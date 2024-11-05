# PhD Template

## Data Preprocessing
All relevant Python scripts are in the directory ```Scripts/Preprocessing/Python```.
- Exploratory data analysis (EDA): ```describe.py```
- Data encoding: ```convert.py```
- Metadata summary: ```metasum.py```
- Setting number of cuts: ```setcut.py```
- Sampling features including instances through SelectKBest: ```selectkbest.py```

## Decision Tree Classifier
- ```Scripts/ML/Python/dtree.py```

## Proposed Mixed Box Classifier
- The most accurate box classifer is built by the CPLEX optimizer.
- The OPL project locates at ```Projects```.
- Important parameters
  - ```cplex.intsollim```: MIP solution number limit (default: 9223372036800000000)
  - ```cplex.threads```: parallel threads (default: 0 implying up to 32 threads)
  - ```cplex.workmem```: working memory before compression and swap (in MB) (default: 2048)
  - ```cplex.trelim```: uncompressed tree memory limit (in MB) (default: 1e+75)
  - ```cplex.nodefileind```: node storage file switch
    - 0 = No node file
    - 1 = Node file in memory and compressed (default)
    - 2 = Node file on disk
    - 3 = Node file on disk and compressed (used in this work)
  - ```cplex.status```: solution status
    - Used for exiting
        - 1 = CPX_STAT_OPTIMAL
        - 101 = CPXMIP_OPTIMAL
        - 102 = CPXMIP_OPTIMAL_TOL
        - 111 = CPXMIP_MEM_LIM_FEAS (Predefined tree memory limit is exceeded and a solution is feasible)
        - 112 = CPXMIP_MEM_LIM_INFEAS
    - Not used for exiting in this work
        - 11 = CPX_STAT_ABORT_TIME_LIM (Accumulated computational time is already recorded in each iteration)
        - 104 = CPXMIP_SOL_LIM (All solutions with different objective values are recorded)
- Execution: ```oplrun -p box 2>&1 | tee <LOG_FILEPATH>```
