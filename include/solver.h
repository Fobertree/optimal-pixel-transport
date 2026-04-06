//
// Created by Alexander Liu on 3/19/26.
//

#ifndef OPTIMALPIXELTRANSPORT_SOLVER_H
#define OPTIMALPIXELTRANSPORT_SOLVER_H

#include "particle_physics.h"
#include "cost_function.h"
#include "consts.h"

// For exp
#include <unsupported/Eigen/MatrixFunctions>

#include <numeric>
#include <algorithm>
#include <random>
#include <type_traits>

// consider templated factory refactor if we want more than one potential solver

class SolverBase {
public:
    explicit SolverBase(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight) {
        src_buf_ = ParticleBuffer(start_path, pWidth, pHeight);
        target_buf_ = ParticleBuffer(target_path, pWidth, pHeight);
    }

    virtual ~SolverBase() = default;

    ParticleBuffer &getSrcBuf() { return src_buf_; }

    [[nodiscard]] const ParticleBuffer &getSrcBuf() const { return src_buf_; }

    // TODO: BENCHMARK THEN FPS UTIL CLASS
    void iterate() {
        // virtual derived solver
        iterateSolver();

        // accumulate collisions
//        ParticlePhysics::computeCollisions(&src_buf_);

        // apply boundaries
        ParticlePhysics::applyBoundaries(&src_buf_);

        // stepPos on each particle
        for (auto &particle: src_buf_) {
            particle->stepPos(DT);
        }

        // debug print
//    printDebug();

        // temp: refresh particle
        // TODO: maybe this can be done via some sort of view?
        src_buf_.refreshParticleCPUBuffer();
    }

    [[nodiscard]] const std::vector<ParticleCPU> &
    getParticleCPUBuffer() const { return src_buf_.getParticleCPUBuffer(); }

protected:
    // declare pure virtual functions
    // Call solver first to accumulate impulse, then call ParticleBuffer::iterate
    virtual void iterateSolver() = 0;

    ParticleBuffer src_buf_, target_buf_;
private:

};

template<typename T, COST_TYPE COST_T = COST_TYPE::RGB_DIST_HYBRID>
class Hungarian : public SolverBase {
    // O(N^2) per iteration O(N^3) total
    // one-to-one bipartite matching
public:
    explicit Hungarian(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight)
            : SolverBase(start_path, target_path, pWidth, pHeight) {
        puts("HUNGARIAN");

        // TODO: cost matrix only in terms of srcLen and tarLen
        len_ = src_buf_.length();

        srcLen_ = static_cast<int>(src_buf_.length());
        tarLen_ = static_cast<int>(target_buf_.length());

        compute_cost_matrix();

        std::cout << std::format("LENGTHS: {} {} {}\n", len_, srcLen_, tarLen_);

        // initial
        job_ = std::vector<int>(tarLen_ + 1, -1);
        ys_ = std::vector<T>(srcLen_ + 1);
        yt_ = std::vector<T>(tarLen_ + 1);

        // reset each iteration
        minTo_ = std::vector<T>(tarLen_ + 1, INF);
        prev_ = std::vector<int>(tarLen_ + 1, -1);
        inZ_ = std::vector<bool>(tarLen_ + 1);

//        std::random_device rd;
//        std::mt19937 g(rd());
//
//        std::shuffle(src_buf_.begin(), src_buf_.end(), g);
    }

    void iterateSolver() override {
        // for each iteration, find the best bipartite matching for src w/ index iter
        if (iter_ >= srcLen_) {
            exert_impulse();
            return;
        }

        int tarCur = tarLen_;
        int srcCur = iter_++;
        job_[tarCur] = srcCur;
        // min reduced cost over edges from Z to src srcCur
        std::fill(minTo_.begin(), minTo_.end(), INF);
        std::fill(prev_.begin(), prev_.end(), -1);
        std::fill(inZ_.begin(), inZ_.end(), false);

        // likely bulk of work
        while (job_[tarCur] != -1) {
            inZ_[tarCur] = true;
            const int src = job_[tarCur];
            T delta = INF;
            int tarNext;
            for (int tar = 0; tar < tarLen_; tar++) {
                // thread pool each iteration as promises - iterations are independent
                if (!inZ_[tar]) {
                    if (ckmin(minTo_[tar], cost_matrix_[src][tar] - ys_[src] - yt_[tar]))
                        prev_[tar] = tarCur;
                    if (ckmin(delta, minTo_[tar]))
                        tarNext = tar;
                }
            }

            // thread pool each iteration as promises - iterations independent
            for (int tar = 0; tar <= tarLen_; tar++) {
                if (inZ_[tar]) {
                    ys_[job_[tar]] += delta;
                    yt_[tar] -= delta;
                } else {
                    minTo_[tar] -= delta;
                }
            }
            tarCur = tarNext;
        }

        // update assignments along alternating path
        for (int tar; tarCur != tarLen_; tarCur = tar) {
            job_[tarCur] = job_[tar = prev_[tarCur]];
        }
        answers_.push_back(-yt_[tarLen_]);

        exert_impulse2();
    }

private:
    void compute_cost_matrix() {
        auto cost_function = get_cost_function<COST_T>();
        len_ = src_buf_.length();

        cost_matrix_ = std::vector<std::vector<float>>(len_, std::vector<float>(len_));

        puts("COST MATRIX INIT");

        // can easily parallelize here + triangulate (if cost function commutative)
        for (int i = 0; i < len_; i++) {
            for (int j = 0; j < len_; j++) {
                auto p1 = src_buf_.getParticle(i);
                auto p2 = target_buf_.getParticle(j);
                cost_matrix_[i][j] = cost_function(*p1, *p2);
            }
        }

        puts("FINISHED COST MATRIX");
    };

    void exert_impulse() {
        // TODO: unroll or parallize this loop
        for (int i = 0; i < tarLen_; i++) {
            int srcIndex = job_[i]; // matched src for target
            auto p1 = src_buf_.getParticle(srcIndex);
            auto pos1 = p1->getPos();
            auto pos2 = target_buf_.getParticle(i)->getPos();

            std::array<float, 2> velo = {pos2[0] - pos1[0], pos2[1] - pos1[1]};

            p1->setVelo(velo);
        }
    }

    void exert_impulse2() {
        // NOTE: augmenting paths can change previous nodes/job_[i]
        // This just creates illusion of real-time
        for (int i = 0; i < tarLen_; i++) {
            int srcIndex = job_[i]; // matched src for target
            if (srcIndex == -1) continue;
            auto p1 = src_buf_.getParticle(srcIndex);
            auto pos1 = p1->getPos();
            auto pos2 = target_buf_.getParticle(i)->getPos();

            std::array<float, 2> velo = {pos2[0] - pos1[0], pos2[1] - pos1[1]};

            p1->setVelo(velo);
        }
    }

    constexpr bool ckmin(T &a, const T &b) {
        return b < a && (a = b, true);
    }

    int iter_ = 0;
    int len_ = -1;
    int srcLen_, tarLen_;

    std::vector<std::vector<float>> cost_matrix_;
    std::vector<T> ys_, yt_, answers_, minTo_;
    std::vector<int> job_, prev_;
    std::vector<bool> inZ_;
    const T INF = std::numeric_limits<T>::max();
};

// Async hungarian: https://web.mit.edu/dimitrib/www/Bertsekas_Castanon_Parallel_Hungarian_1993.pdf

// https://lucyliu-ucsb.github.io/posts/Sinkhorn-algorithm/
template<COST_TYPE T = COST_TYPE::RGB_DIST_HYBRID>
class Sinkhorn : public SolverBase {
public:
    // O(N^2) per iteration
    // eigen
    // Seems numerically unstable depending on regularization param epsilon
    // Want very low epsilon to encourage one-to-one, but this causes kernel to collapse
    explicit Sinkhorn(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight)
            : SolverBase(start_path, target_path, pWidth, pHeight) {

        len_ = src_buf_.length();
        cost_matrix_ = Eigen::MatrixXd(len_, len_);
        compute_cost_matrix();

        A_ = Eigen::VectorXd::Constant(len_, 1. / len_);
        B_ = A_;
        // MUST get ArrayBase for element-wise exp
        // or, will call matrix exponential instead
        K_ = (-cost_matrix_.array() / EPS).exp();   // overflow?

        std::cout << cost_matrix_.maxCoeff() << std::endl;
        std::cout << K_.maxCoeff() << std::endl;

        u_ = Eigen::VectorXd::Ones(len_);
        v_ = u_;

        // P(i,j): i - producer, j - consumer
        P_ = u_.asDiagonal() * K_ * v_.asDiagonal();
        p_norm_ = (P_.transpose() * P_).trace();

        puts("constructor complete");
    }

    void iterateSolver() override {
        // covers one (or N iterations) or Sinkhorn

        u_ = A_.array() / ((K_ * v_).array() + 1e-12).array();
        v_ = B_.array() / ((K_.transpose() * u_).array() + 1e-12).array();

        P_ = u_.asDiagonal() * K_ * v_.asDiagonal();
        // TODO: convert the (P.transpose() * P_).trace() to a function
        // It should be squared matrix L2/Frobenius norm
        if (std::abs(((P_.transpose() * P_).trace() - p_norm_) / p_norm_) < precision_) {
            // solved
            exert_impulse();
            return;
        }
        p_norm_ = (P_.transpose() * P_).trace();

        std::cout << iter_ << std::endl;
//        if (iter_ == 1) std::cout << P_ << std::endl;

        exert_impulse();
        iter_++;
    }

private:
    void compute_cost_matrix() {
        // TODO: move to cost_function or util
        // a cost function

        auto cost_function = get_cost_function<COST_TYPE::RGB_DIST_HYBRID>();

        for (int i = 0; i < len_; i++) {
            auto p1 = src_buf_.getParticle(i);
            for (int j = 0; j < len_; j++) {
                auto p2 = target_buf_.getParticle(j);
                cost_matrix_(i, j) = cost_function(*p1, *p2);
            }
        }

        normalize_cost_matrix();
    }

    void normalize_cost_matrix() {
        double max_cost = cost_matrix_.maxCoeff();
        double min_cost = cost_matrix_.minCoeff();  // or just 0 if costs are non-negative
        cost_matrix_ = (cost_matrix_.array() - min_cost) / (max_cost - min_cost + 1e-8);
    }

    void scale_cost_matrix() {
        cost_matrix_ /= cost_matrix_.mean();
    }

    void exert_impulse() {
        puts("EXERTING IMPULSE");
        // very parallelizable
        for (int i = 0; i < len_; i++) {
            int targetIndex = rowArgmax(i);
            auto p1 = src_buf_.getParticle(i);

            auto pos1 = p1->getPos();
            auto pos2 = target_buf_.getParticle(targetIndex)->getPos();

            // treating as unit mass
            std::array<float, 2> velo = {pos2[0] - pos1[0], pos2[1] - pos1[1]};

            p1->addVelo(velo, 1.);
        }
    }

    [[nodiscard]] int rowArgmax(int i) const {
        // route 'producer' i to best 'consumer' j
        // TODO: always returns 0, fix
        // matrix alarmingly invariant, stops after 1 iteration
        Eigen::MatrixXd::Index maxIndex;
        P_.row(i).maxCoeff(&maxIndex);
//        if (iter_ >= 15)
//            std::cout << "SUM: " << P_.row(i).sum() << std::endl;

        auto maxCoeff = P_.row(i).maxCoeff();
        auto minCoeff = P_.row(i).minCoeff();

        if (iter_ == 5)
            std::cout << std::format("Max: {}, Min: {}, Diff: {}, Index: {}\n", maxCoeff, minCoeff, maxCoeff - minCoeff,
                                     maxIndex);

        return static_cast<int>(maxIndex);
    }

    int iter_ = 0;
    int len_;
    Eigen::MatrixXd cost_matrix_, P_, K_;   // Gibbs kernel (different from kernel in ML, which is similar to RBF but individual weights)
    Eigen::VectorXd A_, B_, u_, v_;
    double p_norm_;
    constexpr static double precision_ = 1e-30;
    constexpr static double EPS = 1e-1;     // regularization param - decrease for more unique
};

class GaleShapley : public SolverBase {
    // proposor: source, acceptor: target
    // problem: sorting everything would be super inefficient O(N^2log N), but very parallelizable
    // bigger problem: proposer optimality (order of proposers does not matter), but no acceptor/rejector optimality
    explicit GaleShapley(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight)
            : SolverBase(start_path, target_path, pWidth, pHeight) {

    }
};

// Genetic algo?

// Jonker-Volgenant
// https://gwern.net/doc/statistics/decision/1987-jonker.pdf
// https://github.com/yongyanghz/LAPJV-algorithm-c/blob/master/src/lap.cpp
template<typename T, COST_TYPE COST_T = COST_TYPE::RGB_DIST_INT_HYBRID>
class LAPJV : public SolverBase {
public:
    // Linear Assignment Problem Jonker-Volgenant
    // Note: LAPJV impl is one-indexed, although particle system is 0-indexed for now
    explicit LAPJV(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight)
            : SolverBase(start_path, target_path, pWidth, pHeight) {
        assert(pWidth == pHeight);
        static_assert(std::is_integral<T>::value, "LAPJV template parameter must be integral");
        n_ = src_buf_.length();
        x_ = std::vector<int>(n_ + 1, 0);
        free_ = std::vector<int>(n_ + 1, -1);
        d_ = std::vector<T>(n_ + 1, 0);
        pred_ = std::vector<int>(n_ + 1, 0);
        y_ = std::vector<int>(n_ + 1, 0);
        v_ = std::vector<T>(n_ + 1, 0);
        u_ = std::vector<T>(n_ + 1, 0);
        col_ = std::vector<int>(n_ + 1, 0);
        last_ = i_ = j_ = -1;

        cost_matrix_ = std::vector<std::vector<T>>(n_ + 1, std::vector<T>(n_ + 1));
        compute_cost_matrix();
        std::iota(col_.begin() + 1, col_.end(), 1);

        // preprocess
        // COLUMN REDUCTION
        for (int j = n_; j > 0; j--) {
            col_[j] = j;
            h_ = cost_matrix_[1][j];
            i1_ = 1;
            for (int i = 2; i <= n_; i++) {
                if (cost_matrix_[i][j] < h_) {
                    h_ = cost_matrix_[i][j];
                    i1_ = i;
                }
            }
            v_[j] = h_;
            if (x_[i1_] == 0) {
                x_[i1_] = j;
                y_[j] = i1_;
            } else {
                x_[i_] = -std::abs(x_[i1_]);
                y_[j] = 0;
            }
        }

        puts("column reduciton ok");

        // REDUCTION TRANSFER
        f_ = 0;
        for (int i = 1; i <= n_; i++) {
            if (x_[i] == 0) {
                f_++;
                free_[f_] = i;
            }
            if (x_[i] < 0) {
                x_[i] = -x_[i];
            } else {
                j1_ = x_[i];
                T min_elem = INF;
                for (int j = 1; j <= n_; j++) {
                    if (j != j1_ && cost_matrix_[i][j] - v_[j] < min_elem) {
                        min_elem = cost_matrix_[i][j] - v_[j];
                    }
                }
                v_[j1_] -= min_elem;
            }
        }

        puts("reduction transfer");

        // AUGMENTING ROW REDUCTION
        for (int cnt = 0; cnt < 2; cnt++) {
            // according to orig. paper, running this twice is empirically optimal
            int k = 1;
            f0_ = f_;
            f_ = 0;
            std::cout << f0_ << std::endl;
            int iters = 0;
            // something wrong here
            while (k <= f0_) {
                if (iters++ > 1000) {
                    puts("too many iters row reduction");
                    std::cout << std::format("k: {}, f0: {}\n", k, f0_);
                    throw std::runtime_error("TOO MANY ITERS IN ROW REDUCTION");
                }
                int i = free_[k];
                k++;
                int u1 = cost_matrix_[i][1] - v_[1];
                j1_ = 1;
                T u2 = INF;

                for (int j = 2; j <= n_; j++) {
                    h_ = cost_matrix_[i][j] - v_[j];
                    if (h_ < u2) {
                        if (h_ >= u1) {
                            u2 = h_;
                            j2_ = j;
                        } else {
                            u2 = u1;
                            u1 = h_;
                            j2_ = j1_;
                            j1_ = j;
                        }
                    }
                }

                i1_ = y_[j1_];
                if (u1 < u2) {
                    v_[j1_] = v_[j1_] - u2 + u1;
                } else if (i1_ > 0) {
                    j1_ = j2_;
                    i1_ = y_[j1_];
                }

                if (i1_ > 0) {
                    if (u1 < u2) {
                        k--;
                        free_[k] = i1_;
                    } else {
                        f_++;
                        free_[f_] = i1_;
                    }
                }
                x_[i] = j1_;
                y_[j1_] = i;
            }
            f0_ = f_;
        }
        f_ = 1; // start for loop from f_ : 1 -> f0
        puts("CONSTRUCTOR OK");
    }

    void iterateSolver() override {
        // this runs the augmentation after pre-processing
        // AUGMENTATION
        if (f_ > f0_) {
            // AUGMENTATION OUTER LOOP DONE
            puts("DONE");
            exert_impulse();
            return;
        }

        i1_ = free_[f_];
        int low = 1, up = 1;
        // can parallelize
        for (j_ = 1; j_ <= n_; j_++) {
            d_[j_] = cost_matrix_[i1_][j_] - v_[j_];
            pred_[j_] = i1_;
        }

        int iters = 0;

        while (true) {
            iters++;
            if (iters > 1000) {
                puts("looping too long in iterateSolver");
                throw std::runtime_error("Looping too long in iterateSolver");
            }
            if (up == low) {
                // fine columns with new value for minimum d
                last_ = low - 1;
                min_elem_ = d_[col_[up]];
                up++;
                for (int k = up; k <= n_; k++) {
                    j_ = col_[k];
                    h_ = d_[j_];
                    if (h_ <= min_elem_) {
                        if (h_ < min_elem_) {
                            up = low;
                            min_elem_ = h_;
                        }
                        col_[k] = col_[up];
                        col_[up] = j_;
                        up++;
                    }
                }
            }
            // can maybe parallelize this via promises/futures
            for (int h = low; h < up; h++) {
                j_ = col_[h];
                // TODO: refactor parts of this into functions, so we can hook on exert_impulse at end
                if (y_[j_] == 0) {
                    augment();
                    exert_impulse();
                    return;
                }
            }
            up = low;
            // scan a row
            j1_ = col_[low];
            low++;
            i_ = y_[j1_];
            int u1 = cost_matrix_[i_][j1_] - v_[j1_] - min_elem_;
            for (int k = up; k <= n_; k++) {
                j_ = col_[k];
                h_ = cost_matrix_[i_][j_] - v_[j_] - u1;
                if (h_ < d_[j_]) {
                    d_[j_] = h_;
                    pred_[j_] = i_;
                    if (h_ == min_elem_) {
                        if (y_[j_] == 0) {
                            augment();
                            std::cout << iters << std::endl;
                            exert_impulse();
                            return;
                        } else {
                            col_[k] = col_[up];
                            col_[up] = j_;
                            up++;
                        }
                    }
                }
            }
        }
    }

private:
    void compute_cost_matrix() {
        // TODO: move to cost_function or util
        // a cost function

        auto cost_function = get_cost_function<COST_T>();

        for (int i = 0; i < n_; i++) {
            auto p1 = src_buf_.getParticle(i);
            for (int j = 0; j < n_; j++) {
                auto p2 = target_buf_.getParticle(j);
                cost_matrix_[i + 1][j + 1] = cost_function(*p1, *p2);
            }
        }
        puts("COST MATRIX OK");
    }

    void exert_impulse() {
//        puts("EXERTING IMPULSE");
        // very parallelizable
        for (int i = 1; i <= n_; i++) {
            int targetIndex = x_.at(i);
            if (targetIndex < 1 || targetIndex > n_) continue;
            // particle buffers are 0-indexed
            auto p1 = src_buf_.getParticle(i - 1);

            auto pos1 = p1->getPos();
            auto pos2 = target_buf_.getParticle(targetIndex - 1)->getPos();

            // treating as unit mass
            std::array<float, 2> velo = {pos2[0] - pos1[0], pos2[1] - pos1[1]};

            p1->addVelo(velo, 1.);
        }
    }

    void augment() {
//        puts("augment");
        for (int k = 1; k <= last_; k++) {
            j1_ = col_[k];
            v_[j1_] = v_[j1_] + d_[j1_] - min_elem_;
        }
        // TODO: double check this i_ initial value
        while (i_ != i1_) {
            i_ = pred_[j_];
            y_[j_] = i_;
            int k = j_;
            j_ = x_[i_];
            x_[i_] = k;
        } // end augmentation

        // DETERMINE ROW PRICES AND OPTIMAL VALUE
        h_ = 0;
        for (i_ = 1; i_ <= n_; i_++) {
            j_ = x_[i_];
            u_[i_] = cost_matrix_[i_][j_] - v_[j_];
            h_ += u_[i_] + v_[j_];
        }
        // h is final price
        f_++;
    }

    // TODO: cost_matrix impl
    std::vector<std::vector<T>> cost_matrix_;
    int n_;  // size
    std::vector<int> x_, y_, free_, pred_;
    std::vector<T> d_;
    std::vector<T> u_, v_;       // duals for reduction - have >=1 elem = 0 in every row + col
    const T INF = std::numeric_limits<T>::max() / 8;

    T h_, min_elem_;
    int f_, i1_, j1_, j2_, f0_, last_{}, i_{}, j_{};

    std::vector<int> col_;
    // LAPJV was originally designed for integer cost matrices
    // in algorithm, we typically assume integer cost matrices
    // in which we can bound worst-case complexity by O(N^3 * R)
    // where R is (integer) range of cost coefficients
    // based on guaranteed reduction by at least 1 in integer case
    // non-integer introduces risk of slower/unstable convergence
    int COST_RESOLUTION_PARAM = 1e4;    // multiply then floor
};

#endif //OPTIMALPIXELTRANSPORT_SOLVER_H
