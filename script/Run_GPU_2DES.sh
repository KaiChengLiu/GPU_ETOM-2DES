#!/bin/sh
#PBS -N Run_CPU_2DES       
#PBS -r n
#PBS -e Run_CPU_2DES.pbserr
#PBS -o Run_CPU_2DES.pbslog
#PBS -q long
#PBS -l walltime=720:00:00,nodes=1:ppn=11

# Set the path to the CPU_2DES program
GPU_2DES=../src/GPU_2DES

# Change to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Change TLIST correspond to your input file
TLIST="0"

# Loop through all combinations of TAU and T
for TAU in $(seq -600 10 600); do
    for T in $TLIST; do
        INPUT_FILE="..2d_input/key_${TAU}_${T}.key"
        OUTPUT_FILE="../2d_output/out_${TAU}_${T}.out"

        # Check if the input file exists
        if [ ! -f "$INPUT_FILE" ]; then
            echo "Input file $INPUT_FILE does not exist!" >> $PBS_O_WORKDIR/error.log
            continue
        fi

        # Execute the program in the background
        ${GPU_2DES} ${INPUT_FILE} > ${OUTPUT_FILE} &

        # Increment the process count
        proc_count=$((proc_count + 1))

        # Check if we've started 11 processes, if so, wait for them to finish
        if [ "$proc_count" -ge 11 ]; then
            wait
            proc_count=0  # Reset the process count after waiting
        fi
    done
done

# Wait for any remaining background processes to finish
wait

echo "All tasks are completed."
