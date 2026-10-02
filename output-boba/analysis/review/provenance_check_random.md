Code of the reruns: commit 6268a4360bbd285bd8b23c97bd854b2cb1df5c9e plus the uncommitted diff of scripts/, the dataset configs and the landscape statistics, sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 over 0 dirty file(s), stamped at 2026-10-02T18:09:45+0100.
No file that a rerun imported changed while the check ran.

| arm | driver | run | status | first differing column (trial or row) | rows differing | max abs diff | verdict |
|---|---|---|---|---|---|---|---|
| output-boba | synthetic | bo_sensor_error_branin_value_random_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 2.84 | exact with acquisition order before qkg/replei were inserted |
| output-boba-confirmatory | synthetic | bo_sensor_error_branin_value_random_seed27_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-extensions | synthetic | bo_sensor_error_bump_a16_w0.05_value_random_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 2.84 | exact with acquisition order before qkg/replei were inserted |
| output-boba-ladder | synthetic | bo_sensor_error_bump_d4_value_random_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 2.84 | exact with acquisition order before qkg/replei were inserted |
| output-boba-misclick | synthetic | bo_sensor_error_branin_value_random_seed7_jittered_exact_misclick_jit20_std0.4.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-mo | synthetic | bo_sensor_error_branincurrin_multi_objective_random_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (41) | 30 of 50 | 3.83 | exact with acquisition order before qkg/replei were inserted |
| output-boba-mo-halo | synthetic | bo_sensor_error_branincurrin_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit20_std1.0_xc0.85.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-mo-halo-backfit | synthetic | bo_sensor_error_branincurrin_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit20_std1.0_xc0.85_halo-backfit.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-slip | synthetic | bo_sensor_error_branin_value_random_seed7_jittered_exact_slip_jit20_std0.4.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-slip-actual | synthetic | bo_sensor_error_branin_value_random_seed7_jittered_exact_slip_jit20_std0.4_rec-actual.csv | exact |  | 0 of 50 | 0 | exact |

Verdicts: exact 6, exact with acquisition order before qkg/replei were inserted 4
