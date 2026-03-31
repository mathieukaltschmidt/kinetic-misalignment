#%%%%%%%%%%%%%%%%%%%%%%%%# grid %
N=128 ; RANKS=2 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device gpu --measCPU"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=0.1   ;   SST0=10  ; LAP=2
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0 --lap $LAP"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=8.0   ;   MSA=1.00   ;   L=6.0    ;   ZEN=5.0   ;   WKB=20.0
PHYS="--qcd qcd --fA 1e8 --msa $MSA --lsize $L  --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
INCO=" --ctype spax --zi 1.0 --sIter 0 --ftype axion --mode0 1"
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
DUMP=100
WTIM=12.0
MEAS=$(echo 1+4+256+16384+65536 | bc ) 
SPMA=$(echo 1 | bc )
SKGV=$(echo 1 | bc )
OUTP="--dump $DUMP --meas $MEAS --p3D 2 --spmask 1 --spKGV 15 --rmask 4.0 --p2Dmap --nologmpi --wTime --p2DmapPE $WTIM"
echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $PREP
echo "         " $OUTP

#export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=$(echo 16/$RANKS | bc)
#USA=" --bind-to socket --mca btl_base_warn_component_unused  0 "

case "$1" in
  create)
    echo "Create run in out/m (or default) and save IC"
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --steps 0 --p3D 1 2>&1 | tee out/log-create.txt
    ;;
  run)
    echo "Run"
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2 2>&1 | tee out/logrun.txt
    ;;
  continue)
    echo "continue Run with index $2 in out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3 2>&1 | tee out/log-continue.txt
    ;;
  con)
    echo "continue Run with index $2 in out/m with extra options $3 in directory $4!"
    mkdir $4 ; mkdir $4/m
    rm $4/m/axion.m.*
    export AXIONS_OUTPUT="$4/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
        cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/$4/m/axion.$find
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --index $2  $3    2>&1 | tee log-con.txt
    ;;
  restart)
    echo "restart Run $AXIONS_OUTPUT/axion.restart"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    WTIM=12
    echo mpirun -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM
    mpirun $USA -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM 2>&1 | tee out/log-restart.txt
    ;;
  redu)
    echo "redo file with index $2 to n = $3"
    echo mpirun -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3
    mpirun $USA -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3
    ;;
  wkb)
    echo "WKB the configuration $AXIONS_OUTPUT/axion.$2 until time --zf $3 in logarithmic --steps $4 "
    mkdir wout ; mkdir wout/m
    rm wout/m/axion.m.*
    export AXIONS_OUTPUT="wout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find
    #echo " ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find"
    mpirun $USA -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --steps $4 --index $2 2>&1 | tee log-wkb.txt
    ;;
  pax)
    echo pax with extra options $4 !
    mkdir pout ; mkdir pout/m
    rm pout/m/axion.*
    export AXIONS_OUTPUT="pout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/wout/m/axion.$find $cdir/pout/m/axion.$find
    mpirun $USA -np $RANKS paxion3d $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --index $2   $4    2>&1 | tee log-pax.txt
    ;;
  measfile)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee log-meas.txt
    ;;

esac
