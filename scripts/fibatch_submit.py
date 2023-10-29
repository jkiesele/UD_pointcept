#!/usr/bin/env python3
"""
Script to submit batch jobs to the FI cluster.
"""

import sys
import os
import uuid

# can be used by others on FI
CONTLOC='/mnt/home/jkieseler/containers/pointcept_add_latest.sif'
UEXT = str(uuid.uuid4())


class meta_option(object):
    def __init__(self, id, default_val = None) -> None:
        self.id = id
        self.triggered = False
        self.value = default_val

    def check(self, clo): 
        if self.triggered:
            self.value = clo
            self.triggered = False
            return True
        if clo == '---'+self.id:
            self.triggered = True
            return True 
        return False
    
    def valid(self):
        return self.value is not None
    
    def __str__(self) -> str:
        return self.id + ': '+self.value
        

opts = {
    'd' : meta_option('d'),
    'n' : meta_option('n',''),
    'c': meta_option('c','a100-80gb'),
    'g': meta_option('g','1'),
    't': meta_option('t','48:00:00'),
}

filtered_clo=[]

for clo in sys.argv:
    next = False
    for _,o in opts.items():
        if o.check(clo):
            next = True
            break
    if next:
        continue
    filtered_clo.append(clo)

all_valid = True
for _,o in opts.items():
    all_valid = all_valid and o.valid()

if '-h' in sys.argv or '--help' in sys.argv or (not all_valid):
    print('script to submit commands within the  container to sbatch.\n')
    print('all commands are fully forwarded with one exception:')
    print('\n    ---d <workdir>    specifies a working directory that can be specified '
      'that will contain the batch logs. It is created if it does not exist.\n')
    print('\n    ---n <name> (opt) specifies a name for the batch script\n')
    print('\n    ---c <constraint> (opt) specifies a resource constraint, default a100-80gb\n')
    print('\n    ---g <number of gpus> \n')
    print('\n    ---t <time> \n')
    sys.exit()

if os.path.isdir(opts['d'].value):
    var = input(\
        'Working directory exists, are you sure you want to continue, please type "yes/y"\n')
    var = var.lower()
    if not var in ('yes', 'y'):
        sys.exit()
else:
    os.system('mkdir -p '+opts['d'].value)


filtered_clo = filtered_clo[1:] #remove
COMMANDS = " "
for clos in filtered_clo:
    COMMANDS += clos + " "

CWD = os.getcwd()

bscript_temp=f'''#!/bin/bash

#SBATCH  -p gpu --gres=gpu:{opts['g'].value}  --mincpus 8 -t {opts['t'].value} --constraint={opts['c'].value}

nvidia-smi
singularity  run  -B /mnt --nv {CONTLOC} /bin/bash {UEXT}_run.sh

'''

runscript_temp=f'''
KTPID=$!
if [[ -f ~/private/wandb_api.sh ]]; then
   source ~/private/wandb_api.sh
fi
cd {CWD}
{COMMANDS}
kill $KTPID
exit
'''

with open(opts['d'].value+'/'+opts['n'].value+'_'+UEXT+'.sh','w', encoding='utf-8') as f:
    f.write(bscript_temp)

with open(opts['d'].value+'/'+UEXT+'_run.sh','w', encoding='utf-8') as f:
    f.write(runscript_temp)

COMMAND = (
    'cd ' + opts['d'].value + '; pwd; module load slurm singularity; unset PYTHONPATH ; '
    'sbatch ' + opts['n'].value + '_' + UEXT + '.sh'
)
os.system(COMMAND)
