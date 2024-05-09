# Microphone frequency response calibration (MFRC)

## Microphone calibration simulation:

run calibration_simulation.py using the following command
```
python calibration_simulation.py -m ism -c -e -p

-m: specify the room type
-c: run calibration
-e: run evaluation
-p: plot figures
```

## DOA simulation:
run DOA_simulation2D_ideal.py using the following command after calibration
```
python DOA_simulation2D_ideal.py
```
