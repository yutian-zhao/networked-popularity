import argparse

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Generate plots of music video network for our work.')
    parser.add_argument( "--title", dest="set_title",
                            action='store_true', 
                            help="Add this flag to show titles in the plots otherwise titles are hidden.")
    parser.add_argument( "--plots", dest="plots",
                            nargs='+',
                            type=int,
                            default=[*range(1,10)],
                            metavar="PLOT_ID", help="Add this flag to specify plots to be generated otherwise all plots are generated. At least one integer from 1 to 9 is required.")    
    args = parser.parse_args()
    return args

def main():
    args = parse_cmd_line_args()
    

if __name__ == '__main__':
    main()
    