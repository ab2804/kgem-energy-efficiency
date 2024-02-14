import argparse
import os

from strep.load_experiment_logs import assemble_database


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir-root", default=r'logs/kgem', help="")
    parser.add_argument("--logdir-merged", default='results_kgem', help="")
    parser.add_argument("--output-tar-dir", default='results_kgem_tar', help="")
    parser.add_argument("--property-extraxtors-module", default='properties_kgem', help="")
    parser.add_argument("--database-dir", default='databases/kgem', help="")

    args = parser.parse_args()

    database = assemble_database(args.logdir_root, args.logdir_merged, args.output_tar_dir, args.property_extraxtors_module)
    if not os.path.isdir(args.database_dir):
        os.makedirs(args.database_dir)
    database.to_pickle(os.path.join(args.database_dir, 'database.pkl'))
