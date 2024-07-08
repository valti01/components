from argparse import ArgumentParser


def main(args):
    print("libary_name", "succeed", "version")
    for libary in args.libs:
        succeed = True
        try:
            m = __import__(libary)
        except:
            succeed = False

        print(libary, succeed, m.__version__ if succeed else None)


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('libs', type=str, nargs='+')
    FLAGS = parser.parse_args()
    main(FLAGS)
