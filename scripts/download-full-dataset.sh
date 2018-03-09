#!/usr/bin/env bash

python ./scripts/download.py -d 00.00 00.01 00.02 00.03 00.04 00.05 00.06 00.07 00.08 00.09 00.10 00.11 01.00 01.01 02.00 \
                        02.01 03.00 04.00 04.01 \
                        00.00.test 00.01.test 01.00.test 01.01.test 02.00.test 02.01.test \
                        03.00.test 04.00.test 04.01.test \
                        $@