#!/bin/bash
for i in {1..20}
do
    let x=i*5-4
    let y=i*5

    let dig1=i/10
    let dig2=i%10

    tmux new-session -s tfrec_$dig1$dig2 -d
    
    tmux send-keys -t tfrec_$dig1$dig2 'python preprocessing.py log-mel-spectrogram /path/to/tfrec/ --tag-path-multi \
                                                                                    /path/to/main.db \
                                                                                    /path/to/extra/database_1.db \
                                                                                    /path/to/extra/database_2.db \
                                                                                    /path/to/extra/database_3.db \
                                                                                    /path/to/extra/database_4.db \
                                                                                    --mels 96 -v -i '
    tmux send-keys -t tfrec_$dig1$dig2 $x ' ' $y
    tmux send-keys -t tfrec_$dig1$dig2 Enter
done
