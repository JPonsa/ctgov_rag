# Set of useful commands

# interactive job
qrsh -pe mpi 1 -l mem=64G,h_rt=2:00:00 -now no
qrsh -pe mpi 8 -l mem=1G,h_rt=1:00:00,gpu=2 -now no

# Request container
apptainer exec --bind /tmp:$TMPDIR ollama.sif ollama run mistral

# Where the HF models are stored
ls /home/rmhijpo/.cache/huggingface/hub/