#!/bin/bash
# Standard-protocol study (Section 4.7 of the paper).
#
# Differences from the controlled protocol of Table 5: ResNet18 with the small-image stem
# (--model resnet18-cifar), epsilon = 8/255, PGD-10 training with step 2/255 and batch size 128, as
# in the adversarial-training literature. CIFAR10: learning rate 0.1, crop-and-flip augmentation,
# 30 epochs. SVHN: learning rate 0.01 (0.1 diverges under adversarial training on SVHN), epsilon
# ramped linearly over the first 5 epochs (at the full radius the model stays at chance), no
# augmentation, 20 epochs. A 5,000-image validation split is held out; PGD-20 and APGD-CE
# (50 iterations) are evaluated on the full test set and AutoAttack on the first 1,000 test images. The main runs keep
# an exponential moving average of the weights (decay 0.999), which is the evaluated model.
#
# Usage: bash experiments/standard_protocol.sh OUT_DIR DATASET RUN SEED [EPOCHS]
#   RUN: at_pgd | at_pgd_sara | at_pgd_matched | pilot_* | at_pgd_sara_evalalign | at_pgd_epN (AT-PGD for N epochs) | at_pgd_sara_epN (SARA for N epochs) | at_pgd_sara_acceptall | inverse_at_like
set -e
OUT=${1:?OUT_DIR}; DS=${2:?DATASET}; RUN=${3:?RUN}; SEED=${4:?SEED}
cd "$(dirname "$0")/.."

case "$DS" in
  cifar10) DSARGS="--lr 0.1"; EPOCHS=${5:-30} ;;
  svhn)    DSARGS="--lr 0.01 --at-eps-warmup 5"; EPOCHS=${5:-20} ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
COMMON="--dataset $DS --model resnet18-cifar --batch-size 128 --weight-decay 5e-4 $DSARGS \
 --base-loss pgd --at-epsilon 0.031373 --pgd-steps 10 --val-size 5000 \
 --eval-every 5 --eval-max-batches 4 --eval-apgd --eval-apgd-iters 50 --eval-autoattack 1000"
# SARA configuration: the variant selected on the CIFAR10 validation split by
# experiments/select_sara_variant.py if the pilot study has been run, otherwise the configuration of Section 4.6.
SEL="$OUT/selected_sara_args.txt"
SARA="--sara $( [ -f "$SEL" ] && cat "$SEL" || echo "--sara-max-correct 64 --sara-align 3 --xi 0.3 --cg-confidence-cap" )"

# Pilot variants (CIFAR10, 10 epochs, no AutoAttack): used only for selection on the validation split.
PILOT_BASE="--sara-max-correct 64 --xi 0.3 --cg-confidence-cap"
PILOT_EPOCHS=10
case "$RUN" in
  pilot_at)          PARGS="" ;;
  pilot_b3)          PARGS="$PILOT_BASE --sara-align 3" ;;
  pilot_b3_eval)     PARGS="$PILOT_BASE --sara-align 3 --sara-align-eval" ;;
  pilot_b6)          PARGS="$PILOT_BASE --sara-align 6" ;;
  pilot_b6_eval)     PARGS="$PILOT_BASE --sara-align 6 --sara-align-eval" ;;
  pilot_b10)         PARGS="$PILOT_BASE --sara-align 10" ;;
  pilot_b3_nobudget) PARGS="--sara-max-correct 0 --xi 0.3 --cg-confidence-cap --sara-align 3" ;;
  pilot_b3_xi05)     PARGS="--sara-max-correct 64 --xi 0.5 --cg-confidence-cap --sara-align 3" ;;
  pilot_b3_k20)      PARGS="$PILOT_BASE --sara-align 3 --kappa 20" ;;
  pilot_b3_xi02)     PARGS="--sara-max-correct 64 --xi 0.2 --cg-confidence-cap --sara-align 3" ;;
  pilot_b3_r2)       PARGS="$PILOT_BASE --sara-align 3 --sara-epsilon 0.062745" ;;
esac

case "$RUN" in
  at_pgd)                EXTRA="--epochs $EPOCHS" ;;
  at_pgd_sara)           EXTRA="--epochs $EPOCHS $SARA" ;;
  at_pgd_sara_ep*)       EXTRA="--epochs ${RUN#at_pgd_sara_ep} $SARA" ;;  # SARA for N epochs (long schedule)
  at_pgd_ep*)            EXTRA="--epochs ${RUN#at_pgd_ep}" ;;
  at_pgd_matched)        # AT-PGD for as long as the selected SARA variant takes (measured cost ratio of the pilots)
                         EXTRA="--epochs $(python -c "print(round($EPOCHS * float(open('$OUT/compute_factor.txt').read())))")" ;;
  at_pgd_sara_evalalign) EXTRA="--epochs $EPOCHS $SARA --sara-align-eval" ;;
  at_pgd_sara_acceptall) EXTRA="--epochs $EPOCHS $SARA --sara-accept-all" ;;
  inverse_at_like)       EXTRA="--epochs $EPOCHS $SARA --sara-max-correct 0 --xi 1.0 --sara-accept-all" ;;
  pilot_at)              EXTRA="--epochs $PILOT_EPOCHS --eval-autoattack 0" ;;
  pilot_*)               EXTRA="--epochs $PILOT_EPOCHS --eval-autoattack 0 --sara $PARGS" ;;
  *) echo "unknown run $RUN"; exit 1 ;;
esac
# Weight averaging (EMA of the weights, decay 0.999) for every run except the selection pilots; it is applied
# identically to all methods, so that it raises the accuracy of the base method and of SARA alike.
case "$RUN" in pilot_*) ;; *) EXTRA="$EXTRA --ema-decay 0.999" ;; esac

# Paths are recorded relative to the repository root, so the released configurations are portable.
DIR="$(realpath -m --relative-to="$PWD" "$OUT/$DS/$RUN/seed$SEED")"
[ -f "$DIR/final.json" ] && { echo "skip $DIR"; exit 0; }
mkdir -p "$DIR"
echo "$(date -u +%H:%M) start $DIR"
echo "$EXTRA" > "$DIR/run_args.txt"
python train.py $COMMON $EXTRA --seed "$SEED" --out-dir "$DIR" > "$DIR/console.log" 2>&1
echo "$(date -u +%H:%M) done  $DIR"
