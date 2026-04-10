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
        // TODO: optimize via diffing approach
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
template<std::integral T, COST_TYPE COST_T = COST_TYPE::RGB_DIST_INT_HYBRID>
class LAPJV : public SolverBase {
public:
    // Linear Assignment Problem Jonker-Volgenant
    // Note: LAPJV impl is one-indexed, although particle system is 0-indexed for now
    // This is highly sequential and might only be faster than Hungarian on GPU
    // In theory, this is supposed to be much faster but so far isn't from experience
    explicit LAPJV(const std::string &start_path, const std::string &target_path, int pWidth, int pHeight)
            : SolverBase(start_path, target_path, pWidth, pHeight) {
        assert(pWidth == pHeight);
        n_ = src_buf_.length();
        // TODO: fire off constructor business logic as an async thread

        cost_matrix_ = std::vector<std::vector<T>>(n_, std::vector<T>(n_));
        compute_cost_matrix();

        puts("OK");

        freeunassigned_ = std::vector<T>(n_);
        collist_ = std::vector<T>(n_);
        matches_ = std::vector<T>(n_, 0);
        d_ = std::vector<T>(n_);
        pred_ = std::vector<T>(n_);
        u_ = std::vector<T>(n_);
        v_ = std::vector<T>(n_);
        rowsol_ = std::vector<T>(n_);
        colsol_ = std::vector<T>(n_, -1);

        puts("Finished alloc");

        f_ = 0; // begin loop

        puts("CONSTRUCTOR DONE");
    }

    void iterateSolver() override {
        if (solver_iter_ == 0) {
            solver_iter_++;
            return;
        }
        if (solver_iter_ == 1) {
            init();
            solver_iter_++;
            f_++;
            return;
        }
//        puts("start solver iter");


        if (f_ >= numfree_) {
            T total_cost = get_total_cost();
//            std::cout << std::format("SOLVED on iter {}: f_: {} >= numfree: {}, total_cost: {}\n", solver_iter_, f_,
//                                     numfree_, total_cost);
            exert_impulse();
            return;
        }
        // augment solution for each free row (1 row per iteration)

        freerow_ = freeunassigned_[f_]; // start row of augmenting path

        // Dijkstra's until unassigned col added to the shortest path tree
        for (j_ = n_; j_--;) {
            d_[j_] = cost_matrix_[freerow_][j_] - v_[j_];
            pred_[j_] = freerow_;
            collist_[j_] = j_; // init column list
        }

        low_ = 0;
        up_ = 0;

        unassignedfound_ = false;
        int iter = 0;
        while (!unassignedfound_) {
            if (up_ == low_) {
                last_ = low_ - 1;
                // scan columns for up..dim-1 to find all indices for which new minimum occurs
                // store these indices between low... up-1 (increasing up)
                min_ = d_[collist_[up_++]];
                for (k_ = up_; k_ < n_; k_++) {
                    j_ = collist_[k_];
                    h_ = d_[j_];
                    if (h_ < min_) {
                        if (h_ < min_) {
                            up_ = low_;
                            min_ = h_;
                        }
                        collist_[k_] = collist_[up_];
                        collist_[up_++] = j_;
                    }
                }
                // check if any of the min columns happen to be unassigned
                // if so, we have an augmenting path right away
                for (k_ = low_; k_ < up_; k_++) {
                    if (colsol_[collist_[k_]] < 0) {
                        endofpath_ = collist_[k_];
                        unassignedfound_ = true;
                        break;
                    }
                }
            } // end (up == low) conditional

            if (!unassignedfound_) {
                // update 'distances' between freerow and all unscanned columns, via next scanned column
                j1_ = collist_[low_];
                low_++;
                i_ = colsol_[j1_];
                h_ = cost_matrix_[i_][j1_] - v_[j1_] - min_;

                for (k_ = up_; k_ < n_; k_++) {
                    j_ = collist_[k_];
                    v2_ = cost_matrix_[i_][j_] - v_[j_] - h_;
                    if (v2_ < d_[j_]) {
                        pred_[j_] = i_;
                        if (v2_ == min_) { // new column found at same minimum value
                            if (colsol_[j_] < 0) {
                                endofpath_ = j_;
                                unassignedfound_ = true;
                                break;
                            } else {
                                collist_[k_] = collist_[up_];
                                collist_[up_++] = j_;
                            }
                        }
                        d_[j_] = v2_; // TODO: CHECK
                    }
                }
            }
        } // end !unassignedfound while loop

        // update column prices
        for (k_ = last_ + 1; k_--;) {
            j1_ = collist_[k_];
            v_[j1_] = v_[j1_] + d_[j1_] - min_;
        }

        // reset row and col assignments along alternating path
        while (i_ != freerow_) {
            i_ = pred_[endofpath_];
            colsol_[endofpath_] = i_;
            j1_ = endofpath_;
            endofpath_ = rowsol_[i_];
            rowsol_[i_] = j1_;
        }

        f_++; // increment for loop
        solver_iter_++;
        exert_impulse();
    }

private:
    void init() {
        // COLUMN REDUCTION
        for (j_ = n_; j_--;) {
            // find min cost over rows
            min_ = cost_matrix_[0][j_];
            imin_ = 0;
            for (i_ = 1; i_ < n_; i_++) {
                if (cost_matrix_[i_][j_] < min_) {
                    min_ = cost_matrix_[i_][j_];
                    imin_ = i_;
                }
            }
            v_[j_] = min_;
            if (++matches_[imin_] == 1) {
                // init assignment if min row assigned for first time
                rowsol_[imin_] = j_;
                colsol_[j_] = imin_;
            } else if (v_[j_] < v_[rowsol_[imin_]]) {
                int j1 = rowsol_[imin_];
                rowsol_[imin_] = j_;
                colsol_[j_] = imin_;
                colsol_[j1] = -1;
            } else
                colsol_[j_] = -1; // row already assigned, column not assigned
        }

        puts("REDUCTION TRANSFER");

        // REDUCTION TRANSFER
        for (i_ = 0; i_ < n_; i_++) {
            if (matches_[i_] == 0)
                freeunassigned_[numfree_++] = i_;
            else if (matches_[i_] == 1) { //transfer reduction from rows that are assigned once
                j1_ = rowsol_[i_];
                min_ = INF;
                for (j_ = 0; j_ < n_; j_++) {
                    if (j_ != j1_ && cost_matrix_[i_][j_] - v_[j_] < min_)
                        min_ = cost_matrix_[i_][j_] - v_[j_];
                }
                if (min_ < 0) {
                    std::cerr << "ERROR IN REDUCTION TRANSFER::" << min_ << std::endl;
                }
                v_[j1_] -= min_;
            }
        }

        for (int loopcnt = 0; loopcnt < 2; loopcnt++) {
            // original paper says running this routine twice is optimal
            k_ = 0;
            prvnumfree_ = numfree_;
            numfree_ = 0;
            int iters = 0;
            while (k_ < prvnumfree_) {
                iters++;

                i_ = freeunassigned_[k_];
                k_++;
                // find minimum and second minimum reduced cost over columns
                umin_ = cost_matrix_[i_][0] - v_[0];
                j1_ = 0;
                usubmin_ = INF;

                for (j_ = 1; j_ < n_; j_++) {
                    h_ = cost_matrix_[i_][j_] - v_[j_];
                    if (h_ < usubmin_) {
                        if (h_ >= umin_) {
                            usubmin_ = h_;
                            j2_ = j_;
                        } else {
                            usubmin_ = umin_;
                            umin_ = h_;
                            j2_ = j1_;
                            j1_ = j_;
                        }
                    }
                }
                i0_ = colsol_[j1_];
                if (umin_ < usubmin_) {
                    v_[j1_] = v_[j1_] - (usubmin_ - umin_);
                } else if (i0_ >= 0) {
                    j1_ = j2_;
                    i0_ = colsol_[j2_];
                }
                // (re-)assign i to j1, possibly de-assigning an i0
                rowsol_[i_] = j1_;
                colsol_[j1_] = i_;

                if (i0_ >= 0) {
                    if (umin_ < usubmin_)
                        freeunassigned_[--k_] = i0_;
                    else
                        freeunassigned_[numfree_++] = i0_;
                }

                exert_impulse();
            } // end while loop
//            std::cout << "iters: " << iters << std::endl;
        } // end augment row reduction subroutine
        f_ = 0;

//        std::cout << "NUMFREE: " << numfree_ << std::endl;
        puts("PRECOMPUTE DONE");
    }

    void compute_cost_matrix() {
        // TODO: move to cost_function or util
        // a cost function

        auto cost_function = get_cost_function<COST_T>();

        for (int i = 0; i < n_; i++) {
            auto p1 = src_buf_.getParticle(i);
            for (int j = 0; j < n_; j++) {
                auto p2 = target_buf_.getParticle(j);
                cost_matrix_[i][j] = cost_function(*p1, *p2);
                if (cost_matrix_[i][j] >= INF) throw std::runtime_error("INF TOO SMALL");
            }
        }
        puts("COST MATRIX OK");
    }

    void exert_impulse() {
//        puts("EXERTING IMPULSE");
        // very parallelizable
        std::vector<T> debug_rowsol(rowsol_.begin(), rowsol_.end());
//        std::sort(debug_rowsol.begin(), debug_rowsol.end());
//        std::copy(debug_rowsol.begin(), debug_rowsol.end(), std::ostream_iterator<T>(std::cout, " "));
        std::vector<T> debug_colsol(colsol_.begin(), colsol_.end());
//        for (auto dbg: debug_colsol) {
//            std::cout << dbg << " ";
//        }
//        puts("");
        for (int i = 0; i < n_; i++) {
            int targetIndex = rowsol_.at(i);
            if (targetIndex < 1 || targetIndex > n_) continue;
            // particle buffers are 0-indexed
            auto p1 = src_buf_.getParticle(i);

            auto pos1 = p1->getPos();
            auto pos2 = target_buf_.getParticle(targetIndex)->getPos();

            // treating as unit mass
            std::array<float, 2> velo = {pos2[0] - pos1[0], pos2[1] - pos1[1]};

            p1->setPos(pos2[0], pos2[1]);
        }
    }

    T get_total_cost() {
        T lapcost = 0;
        for (int i = n_; i--;) {
            int j = rowsol_[i];
            u_[i] = cost_matrix_[i][j] - v_[j];
            lapcost = lapcost + cost_matrix_[i][j];
        }
        return lapcost;
    }

    std::vector<std::vector<T>> cost_matrix_;
    int n_;  // size
    bool unassignedfound_;
    T i_, imin_, numfree_ = 0, prvnumfree_, f_, i0_, k_, freerow_;
    std::vector<T> pred_, freeunassigned_;
    T j_, j1_, j2_, endofpath_, last_, low_, up_;
    std::vector<T> collist_, matches_;
    T min_, h_, umin_, usubmin_, v2_;
    std::vector<T> d_, rowsol_, colsol_, u_, v_;

    T INF = std::numeric_limits<T>::max() / 2; // div by 2 prevent underflow
    uint64_t solver_iter_ = 0;

    // LAPJV was originally designed for integer cost matrices
    // in algorithm, we typically assume integer cost matrices
    // in which we can bound worst-case complexity by O(N^3 * R)
    // where R is (integer) range of cost coefficients
    // based on guaranteed reduction by at least 1 in integer case
    // non-integer introduces risk of slower/unstable convergence
};

#endif //OPTIMALPIXELTRANSPORT_SOLVER_H
