import CMB_PAInT as CMBP

Cls = "./../examples/docs/Cls_TT_ns_16_fwhm_640.npy"
Nside = 16
Lmax = 48
Out_PATH = "./test_intensity/"
Inp_In_PATH = "./../examples/docs/"
Inp_Out_PATH = "./test_intensity/"
In_Name = "example_map_T_adding_reg_noise.fits"
Num_Sims = 1
Single = True
No_Z = False
Cons_Uncons = True
Fields = 3
Zbar_PATH = None
I_Mask = "./../examples/docs/example_masks/example_I_mask.fits"
Chunks = 96
job_name = None
env = None
Noise_Level = 0.00001
DP = True
local = True
ntasks_cov = 4
P_Mask = None
ntasks_inp = 4

Global_Path = "./../CMBPAInT/"

Inpainting_Data = CMBP.CMB_PAInT(Cls, Nside, Lmax, Out_PATH, Inp_In_PATH, Inp_Out_PATH, In_Name, Num_Sims, Single, No_Z, Cons_Uncons, Fields, Zbar_PATH, I_Mask, Chunks, job_name, env, Noise_Level = Noise_Level, DP = DP, local = local, ntasks_cov = ntasks_cov, P_Mask = P_Mask, ntasks_inp = ntasks_inp)

Inpainting_Data.generate_config_file()

Inpainting_Data.calculate_covariance_matrix(global_path = Global_Path)

Inpainting_Data.calculate_chol(global_path = Global_Path)

Inpainting_Data.calculate_inp(global_path = Global_Path)

print("Test completed...")
