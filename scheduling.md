Scheduling:

We have tried these schedules so far:
The metrics are stored in the corresponding csv files, e.g. test_metrics_adaptive_schedule_2.csv

IMPORTANT:
Before running a new schedule x, change the name of schedule_name in train.py, so the metrics are saved automatically in "test_metrics_{schedule_name}.csv":

with open("test_metrics_adaptive_schedule_x.csv", "a") as f:
                    f.write(f"{iteration},{psnr_test},{ssim_test},{lpips_test}\n")

'''schedule 1:
    first_phase_start = 5000
    second_phase_start = 10000
    third_phase_start = 20000
    third_phase_end = opt.iterations - 1 
    average_gradients_over = 100 
    first_phase_ratio = 0.005
    second_phase_ratio = 0.01
    third_phase_ratio = 0.002
    first_phase_frequency = 500
    second_phase_frequency = 500
    third_phase_frequency = 1000

    first_phase_max_degree = 1
    second_phase_max_degree = 3
    third_phase_max_degree = 3

    '''
    '''
    schedule 2 (best so far):
    first_phase_start = 2000 # standard: 5000
    second_phase_start = 6000 # standard: 10000
    third_phase_start = 12000 # standard: 20000
    third_phase_end = opt.iterations - 1 # standard: opt.iterations - 1
    average_gradients_over = 100 
    first_phase_ratio = 0.01 # standard: 0.005
    second_phase_ratio = 0.015 # standard: 0.01
    third_phase_ratio = 0.005 # standard: 0.002
    first_phase_frequency = 250 # standard: 500
    second_phase_frequency = 250 # standard: 500
    third_phase_frequency = 1500 # standard: 1000

    first_phase_max_degree = 1
    second_phase_max_degree = 3
    third_phase_max_degree = 3
    '''



    # schedule 3 (slightly under schedule 2):
    first_phase_start  = 2000
    second_phase_start = 6000
    third_phase_start  = 12000
    third_phase_end    = opt.iterations - 1

    average_gradients_over = 50   # ↓ faster reaction

    first_phase_ratio  = 0.01    # ↑ slightly more early capacity
    second_phase_ratio = 0.015    # ↑ main expressive window
    third_phase_ratio  = 0.001    # ↓↓↓ strong taper (key change)

    first_phase_frequency  = 250
    second_phase_frequency = 250
    third_phase_frequency  = 2000 # ↓ late churn

    first_phase_max_degree = 1
    second_phase_max_degree = 2
    third_phase_max_degree = 3

    Next step: Introduce percentage of increase for each SH degree, i.e. the top x% of gaussians with SH degree 0 get updated, the top y% of gaussians with SH degree 1 get updated,...

    Idea for schedule 4: Add a lot of capacity for SH degree 1 and 2 already at the start, don't allow SH degree 3 until third phase

    # schedule 4:
    schedule_name = "adaptive_schedule_4"
    average_gradients_over = 50
    first_phase_start = 1000
    second_phase_start = 5000
    third_phase_start = 15000

    first_phase_ratio = 0.01
    second_phase_ratio_for_degree_2 = 0.01
    second_phase_ratio_for_degree_1 = 0.01
    second_phase_ratio_for_degree_0 = 0.0025
    third_phase_ratio = 0.001

    first_phase_frequency = 250
    second_phase_frequency = 250
    third_phase_frequency = 1500

    first_phase_max_degree = 2
