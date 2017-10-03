# Demonstrates how to use adversarial example (AE) codes.
#
# We have had so many issues with our custom codes/docker images for the contest that 
# we just fell back to using a minor variant of the example codes provided by the 
# competition organizers.  Had we anticipated the difficulties (compatability with keras,
# issues with docker running remotely) we would have done this from the outset...
#

# mjp, september 2017


#-------------------------------------------------------------------------------
# System-dependent paths
#-------------------------------------------------------------------------------

DATA_DIR=${HOME}/Data//NIPS2017/images
VALID_DIR=${HOME}/Documents/cleverhans/examples/nips17_adversarial_competition/validation_tool

LOCAL_DATA=./NIPS_1000

#-------------------------------------------------------------------------------
# The rest of this file should be system-independent
#-------------------------------------------------------------------------------

TARGETED_ZIP_FILE=targeted-iter-gd-ens.zip
ATTACK_ZIP_FILE=non-targeted-iter-gd-ens.zip
DEFENSE_ZIP_FILE=defense.zip

# perturbation size, in \ell_\infty
TAU=16


#-------------------------------------------------------------------------------
# Targets for running local experiments
# For conventional tasks, we use the run_*.sh scripts (vs. calling python directly)
#-------------------------------------------------------------------------------

# generate a train/test split for our experiments
$(LOCAL_DATA) :
	python split_train_test.py $(DATA_DIR) 900


demo-attack : $(LOCAL_DATA)
	\rm -rf ./Output_Attack && mkdir -p ./Output_Attack 
	time ./run_attack.sh $(LOCAL_DATA)/Test ./Output_Attack $(TAU)


demo-targeted : $(LOCAL_DATA)
	\rm -rf ./Output_Targeted && mkdir -p ./Output_Targeted 
	time ./run_attack.sh $(LOCAL_DATA)/Test_Targeted ./Output_Targeted $(TAU)


# run this 1x before running demo-defense
baseline-defense :
	\rm -rf ./Baselines && mkdir -p ./Baselines
	python defense.py --input_dir=$(LOCAL_DATA)/Train --output_file=""


# run an untargeted attack first (to create ./Output_Attack)
demo-defense : ./Output_Attack
	time ./run_defense.sh ./Output_Attack 



#-------------------------------------------------------------------------------
# Targets for creating and validating zip files
#-------------------------------------------------------------------------------

zip : zip-targeted zip-attack zip-defense


zip-targeted :
	\rm -f $(TARGETED_ZIP_FILE) 
	cp metadata.targeted metadata.json
	zip $(TARGETED_ZIP_FILE) Makefile metadata.json run_attack.sh ./*.py ./Weights/* ./ens_adv_inception_resnet_v2/*
	\rm -f metadata.json

zip-attack :
	\rm -f $(ATTACK_ZIP_FILE) 
	cp metadata.non-targeted metadata.json
	zip $(ATTACK_ZIP_FILE) Makefile metadata.json run_attack.sh ./*.py ./Weights/* ./ens_adv_inception_resnet_v2/*
	\rm -f metadata.json

zip-defense :
	\rm -f $(DEFENSE_ZIP_FILE) 
	cp metadata.defense metadata.json
	zip $(DEFENSE_ZIP_FILE) Makefile metadata.json run_defense.sh ./*.py ./Weights/* ./ens_adv_inception_resnet_v2/* ./Baselines/*.npy
	\rm -f metadata.json


# deletes zip files
zip-clean :
	\rm -f $(DEFENSE_ZIP_FILE) $(ATTACK_ZIP_FILE) $(TARGETED_ZIP_FILE) metadata.json


# Runs the competition's validation tool on our zip file.
# You will need to run "make zip" first...
validate-targeted : 
	PYTHONPATH=$(VALID_DIR) python $(VALID_DIR)/validate_submission.py \
		--submission_filename $(TARGETED_ZIP_FILE) \
		--submission_type targeted_attack \
		--use_gpu

validate-attack : 
	PYTHONPATH=$(VALID_DIR) python $(VALID_DIR)/validate_submission.py \
		--submission_filename $(ATTACK_ZIP_FILE) \
		--submission_type attack \
		--use_gpu


validate-defense : 
	PYTHONPATH=$(VALID_DIR) python $(VALID_DIR)/validate_submission.py \
		--submission_filename $(DEFENSE_ZIP_FILE) \
		--submission_type defense \
		--use_gpu


#-------------------------------------------------------------------------------
# targets for specialized side-experiments
#-------------------------------------------------------------------------------

attack-single-network :
	mkdir -p ./Output-vs-AdvInceptionV3
	python attack_iter_target_class.py --input_dir=$(DATA_DIR) --output_dir=./Output-vs-AdvInceptionV3 --max_epsilon=8 --num_iter=10 --target_model=Adv-InceptionV3
	echo ''
	mkdir -p ./Output-vs-InceptionV3
	python attack_iter_target_class.py --input_dir=$(DATA_DIR) --output_dir=./Output-vs-InceptionV3 --max_epsilon=8 --num_iter=10 --target_model=InceptionV3
	echo ''
	mkdir -p ./Output-vs-ResnetV2
	python attack_iter_target_class.py --input_dir=$(DATA_DIR) --output_dir=./Output-vs-ResnetV2 --max_epsilon=8 --num_iter=10 --target_model=resnet_v2_101
	echo ''
	mkdir -p ./Output-vs-InceptionResnetV2
	python attack_iter_target_class.py --input_dir=$(DATA_DIR) --output_dir=./Output-vs-InceptionResnetV2 --max_epsilon=8 --num_iter=10 --target_model=InceptionResnetV2
