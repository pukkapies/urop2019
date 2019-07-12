import os, sys

if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../modules')))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

def die_with_usage():
    print()
    print("Mp3Wrangler - Script to fetch Mp3 songs from the server and output an ultimate CSV file")
    print()
    print("Usage:     python data_wrangle.py <csv filename or path> [options]")
    print()
    print("General Options:")
    print("  --discard-no-tag         Choose to discard tracks with no tags.")
    print("  --discard-duplic <mode>  Choose to discard duplicate tracks.") # <mode> not currently supported
    print("  --help                   Show this help message and exit.")
    print("  --threshold              Set the minimum size (in bytes) to allow for the MP3 files (default 0).")
    print()
    print("Example:   python data_wrangle.py ./wrangl.csv --threshold 50000 --discard-no-tag")
    print()
    sys.exit(0)

if __name__ == "__main__":

    from wrangler import ultimate_output 

    # show help
    if len(sys.argv) < 2:
        die_with_usage()
    
    # show help, if user did not input something weird
    if '--help' in sys.argv:
        if len(sys.argv) == 2:
            die_with_usage()
        else:
            print('???')
            sys.exit(0)
    
    if sys.argv[1][-4:] == '.csv':
        output = sys.argv[1]
    else:
        output = sys.argv[1] + '.csv'

    # check arguments
    if len(sys.argv) == 2:
        df = ultimate_output()
        df.to_csv(output, index=False)
    else:
        # initialize variables
        threshold = 0 
        discard_no_tag = False
        discard_duplic = False

        while True:
            if len(sys.argv) == 2:
                break
            elif sys.argv[2] == '--threshold':
                threshold = int(sys.argv[3])
                del sys.argv[2:4]
            elif sys.argv[2] == '--discard-no-tag':
                discard_no_tag = True
                del sys.argv[2]
            elif sys.argv[2] == '--discard-duplic':
                discard_duplic = True
                del sys.argv[2]            
            else:
                print('???')
                sys.exit(0)

        df = ultimate_output(threshold, discard_no_tag, discard_dupl)
        df.to_csv(output, index=False)