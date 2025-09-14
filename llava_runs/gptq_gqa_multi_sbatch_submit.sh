python multi_sbatch_gptq_gqa.py --env slurm_files \
                                --nhrs 4 \
                                --qos scav \
                                --partition vulcan \
                                --gpu 1 \
                                --gpu-type a5000 a6000 \
                                --cores 1 \
                                --mem 48 \
                                --output-dirname gpa_gptq_output \
                                # --dryrun
                                

