//
// CPU LAPJV (Jonker-Volgenant) — reference port from lap.cpp
//

#ifndef OPTIMALPIXELTRANSPORT_LAPJV_CPU_H
#define OPTIMALPIXELTRANSPORT_LAPJV_CPU_H

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lapjv_cpu {

inline int32_t cost_at(const std::vector<int32_t> &cost, int n, int row, int col) {
    return cost[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)];
}

// Returns rowsol[row] = assigned column for each row (0-indexed permutation).
inline std::vector<int32_t> solve(const std::vector<int32_t> &cost, int n) {
    if (n <= 0) {
        return {};
    }
    const size_t expected = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (cost.size() != expected) {
        throw std::runtime_error("LAPJV cost matrix size mismatch");
    }

    const int32_t INF = std::numeric_limits<int32_t>::max() / 2;

    std::vector<int32_t> rowsol(static_cast<size_t>(n));
    std::vector<int32_t> colsol(static_cast<size_t>(n), -1);
    std::vector<int32_t> u(static_cast<size_t>(n));
    std::vector<int32_t> v(static_cast<size_t>(n));
    std::vector<int32_t> pred(static_cast<size_t>(n));
    std::vector<int32_t> free_rows(static_cast<size_t>(n));
    std::vector<int32_t> collist(static_cast<size_t>(n));
    std::vector<int32_t> matches(static_cast<size_t>(n), 0);
    std::vector<int32_t> d(static_cast<size_t>(n));

    // COLUMN REDUCTION
    for (int j = n - 1; j >= 0; --j) {
        int32_t min_c = cost_at(cost, n, 0, j);
        int imin = 0;
        for (int i = 1; i < n; ++i) {
            const int32_t c = cost_at(cost, n, i, j);
            if (c < min_c) {
                min_c = c;
                imin = i;
            }
        }
        v[static_cast<size_t>(j)] = min_c;
        if (++matches[static_cast<size_t>(imin)] == 1) {
            rowsol[static_cast<size_t>(imin)] = j;
            colsol[static_cast<size_t>(j)] = imin;
        } else if (min_c < v[static_cast<size_t>(rowsol[static_cast<size_t>(imin)])]) {
            const int j1 = rowsol[static_cast<size_t>(imin)];
            rowsol[static_cast<size_t>(imin)] = j;
            colsol[static_cast<size_t>(j)] = imin;
            colsol[static_cast<size_t>(j1)] = -1;
        } else {
            colsol[static_cast<size_t>(j)] = -1;
        }
    }

    // REDUCTION TRANSFER
    int numfree = 0;
    for (int i = 0; i < n; ++i) {
        if (matches[static_cast<size_t>(i)] == 0) {
            free_rows[static_cast<size_t>(numfree++)] = i;
        } else if (matches[static_cast<size_t>(i)] == 1) {
            const int j1 = rowsol[static_cast<size_t>(i)];
            int32_t min_red = INF;
            for (int jj = 0; jj < n; ++jj) {
                if (jj != j1) {
                    min_red = std::min(min_red, cost_at(cost, n, i, jj) - v[static_cast<size_t>(jj)]);
                }
            }
            v[static_cast<size_t>(j1)] -= min_red;
        }
    }

    // AUGMENTING ROW REDUCTION (2 passes)
    for (int loopcnt = 0; loopcnt < 2; ++loopcnt) {
        int k = 0;
        const int prvnumfree = numfree;
        numfree = 0;
        while (k < prvnumfree) {
            const int i = free_rows[static_cast<size_t>(k++)];

            int32_t umin = cost_at(cost, n, i, 0) - v[0];
            int j1 = 0;
            int32_t usubmin = INF;
            int j2 = 1;

            for (int jj = 1; jj < n; ++jj) {
                const int32_t h = cost_at(cost, n, i, jj) - v[static_cast<size_t>(jj)];
                if (h < usubmin) {
                    if (h >= umin) {
                        usubmin = h;
                        j2 = jj;
                    } else {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = jj;
                    }
                }
            }

            int i0 = colsol[static_cast<size_t>(j1)];
            if (umin < usubmin) {
                v[static_cast<size_t>(j1)] -= (usubmin - umin);
            } else if (i0 >= 0) {
                j1 = j2;
                i0 = colsol[static_cast<size_t>(j2)];
            }

            rowsol[static_cast<size_t>(i)] = j1;
            colsol[static_cast<size_t>(j1)] = i;

            if (i0 >= 0) {
                if (umin < usubmin) {
                    free_rows[static_cast<size_t>(--k)] = i0;
                } else {
                    free_rows[static_cast<size_t>(numfree++)] = i0;
                }
            }
        }
    }

    // AUGMENTATION FOR EACH FREE ROW
    for (int f = 0; f < numfree; ++f) {
        const int freerow = free_rows[static_cast<size_t>(f)];

        for (int jj = n - 1; jj >= 0; --jj) {
            d[static_cast<size_t>(jj)] = cost_at(cost, n, freerow, jj) - v[static_cast<size_t>(jj)];
            pred[static_cast<size_t>(jj)] = freerow;
            collist[static_cast<size_t>(jj)] = jj;
        }

        int low = 0;
        int up = 0;
        int last = -1;
        bool unassignedfound = false;
        int endofpath = 0;
        int32_t min_d = 0;

        while (!unassignedfound) {
            if (up == low) {
                last = low - 1;
                min_d = d[static_cast<size_t>(collist[static_cast<size_t>(up++)])];
                for (int kk = up; kk < n; ++kk) {
                    const int jj = collist[static_cast<size_t>(kk)];
                    const int32_t h = d[static_cast<size_t>(jj)];
                    if (h <= min_d) {
                        if (h < min_d) {
                            up = low;
                            min_d = h;
                        }
                        collist[static_cast<size_t>(kk)] = collist[static_cast<size_t>(up)];
                        collist[static_cast<size_t>(up++)] = jj;
                    }
                }
                for (int kk = low; kk < up; ++kk) {
                    if (colsol[static_cast<size_t>(collist[static_cast<size_t>(kk)])] < 0) {
                        endofpath = collist[static_cast<size_t>(kk)];
                        unassignedfound = true;
                        break;
                    }
                }
            }

            if (!unassignedfound) {
                const int j1 = collist[static_cast<size_t>(low++)];
                const int i_path = colsol[static_cast<size_t>(j1)];
                const int32_t h = cost_at(cost, n, i_path, j1) - v[static_cast<size_t>(j1)] - min_d;

                for (int kk = up; kk < n; ++kk) {
                    const int jj = collist[static_cast<size_t>(kk)];
                    const int32_t v2 = cost_at(cost, n, i_path, jj) - v[static_cast<size_t>(jj)] - h;
                    if (v2 < d[static_cast<size_t>(jj)]) {
                        pred[static_cast<size_t>(jj)] = i_path;
                        if (v2 == min_d) {
                            if (colsol[static_cast<size_t>(jj)] < 0) {
                                endofpath = jj;
                                unassignedfound = true;
                                break;
                            }
                            collist[static_cast<size_t>(kk)] = collist[static_cast<size_t>(up)];
                            collist[static_cast<size_t>(up++)] = jj;
                        }
                        d[static_cast<size_t>(jj)] = v2;
                    }
                }
            }
        }

        for (int k = last + 1; k-- > 0;) {
            const int j1 = collist[static_cast<size_t>(k)];
            v[static_cast<size_t>(j1)] += d[static_cast<size_t>(j1)] - min_d;
        }

        int i_path = 0;
        do {
            i_path = pred[static_cast<size_t>(endofpath)];
            colsol[static_cast<size_t>(endofpath)] = i_path;
            const int j1 = endofpath;
            endofpath = rowsol[static_cast<size_t>(i_path)];
            rowsol[static_cast<size_t>(i_path)] = j1;
        } while (i_path != freerow);
    }

    return rowsol;
}

} // namespace lapjv_cpu

#endif // OPTIMALPIXELTRANSPORT_LAPJV_CPU_H