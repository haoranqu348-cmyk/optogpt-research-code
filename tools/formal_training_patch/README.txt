OptoGPT joint s+p production-training patch

Files:
  joint_sp/scripts/finetune.py
  joint_sp/model.py
  build_formal_data.ps1
  extend_formal_joint_data.ps1

Apply on Windows project root:
  D:\hrqu\optogpt_project\optogpt

IMPORTANT:
1. Back up the two existing files with a timestamp before copying.
2. This patch keeps CPU checkpoint loading for RNG restoration.
3. New checkpoints include scheduler_state_dict, scaler_state_dict,
   batches_per_epoch, and batch-level global_step.
4. Old smoke checkpoints do not contain scheduler/scaler state and must
   not be resumed by this version. Re-run smoke training from model\\optogpt.pt.
5. Run the full unit tests, then a fresh one-epoch smoke and fresh resume
   before any large data generation or formal training.
6. build_formal_data.ps1 refuses to overwrite its p-polarization anchor or
   joint output directory. Its default joint output is:
     D:\hrqu\optogpt_project\optogpt\data_60deg_sp_joint_500k_v1
7. The formal-data build reuses the verified 500k s-polarization structures
   and computes matching p spectra with TMM for those same structures. The
   published layout is [Rs(71), Ts(71), Rp(71), Tp(71)].

Suggested commands after copying:
  python -m py_compile joint_sp\\scripts\\finetune.py joint_sp\\model.py
  python -m unittest discover -s joint_sp\\tests -p "test*.py" -v

Formal-data build (large operation, only after approval):
  powershell -ExecutionPolicy Bypass -File .\\build_formal_data.ps1

Resume an interrupted build with the same Workers and ChunkSize:
  powershell -ExecutionPolicy Bypass -File .\\build_formal_data.ps1 -Resume

Do not start formal training unless the script ends with:
  PREFLIGHT_OK
  FORMAL_JOINT_DATA_OK

If the v1 build reports exactly 252007 unique structures, extend it without
overwriting v1:
  powershell -ExecutionPolicy Bypass -File .\\extend_formal_joint_data.ps1

Resume an interrupted v2 extension with the original arguments:
  powershell -ExecutionPolicy Bypass -File .\\extend_formal_joint_data.ps1 -Resume

The v2 output is:
  D:\hrqu\optogpt_project\optogpt\data_60deg_sp_joint_500k_v2

Do not train unless the v2 script ends with:
  EXTENDED_FORMAL_JOINT_DATA_OK
  PREFLIGHT_OK
  FORMAL_V2_PREFLIGHT_AND_TMM_OK
