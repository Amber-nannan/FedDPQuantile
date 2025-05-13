# nohup python case_ray_tune_2.py > nohup_r_zero_point_nine.out 2>&1 &

# nohup python case_ray_tune.py > nohup_add_C.out 2>&1 &


# 顺序执行两个实验，第一个完全运行完成后再启动第二个
# nohup python case_ray_tune.py > nohup_sigma_smaller.out 2>&1 && \
# nohup python case_ray_tune_2.py > nohup_r_one.out 2>&1 &


# nohup python case_ray_tune_2.py -r 0.5 > nohup_r_zero_point_five.out 2>&1 && \
# nohup python case_ray_tune_2.py -r 0.85 > nohup_r_zero_point_eight_five.out 2>&1 && \
# nohup python case_ray_tune_2.py -r 0.9 > nohup_r_zero_point_nine.out 2>&1 && \
# nohup python case_ray_tune_2.py -r 0.95 > nohup_r_zero_point_nine_five.out 2>&1 &


# nohup python case_ray_tune_3.py -E log > nohup_E_log.out 2>&1 &

# nohup python case_ray_tune_3.py -E 1 > nohup_E_one.out 2>&1 && \
# nohup python case_ray_tune_3.py -E 5 > nohup_E_five.out 2>&1 &



nohup python case_ray_tune_4.py -r 0.25 -E 1 > nohup_r_0_25_E_1.out 2>&1 &
nohup python case_ray_tune_4.py -r 0.9 -E log > nohup_r_0_9_E_log.out 2>&1 &


nohup python case_ray_tune_4.py -r 0.9 -E log > nohup_r_0_9_E_log_c_1.out 2>&1 &
nohup python case_ray_tune_4_copy.py -r 0.9 -E log > nohup_r_0_9_E_log_c_2.out 2>&1 &

nohup python case_ray_tune_4.py -r 0.25 -E 1 > nohup_r_0_25_E_log_c_1.out 2>&1 &
nohup python case_ray_tune_4_copy.py -r 0.25 -E 1 > nohup_r_0_25_E_log_c_2.out 2>&1 &