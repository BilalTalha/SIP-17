# Train:

python ConvNext_train_all.py --data_path .../ISD_dataset/Data_with_texture/All_objects --output_dir .../ISD/ConvNeXt/output/all_objects/trainbase1_25.pth


# Test:
python ConvNext_test_all.py --data_path .../ISD_dataset/Test_final/All_objects --weights ".../ISD/ConvNext/output/all_objects/trainbase1_25.pth"
