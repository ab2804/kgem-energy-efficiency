import argparse
import traceback
from datetime import timedelta
import json
import os
import time
import random

from codecarbon import OfflineEmissionsTracker

from mlprops.util import create_output_dir, PatchedJSONEncoder

from pykeen.pipeline import pipeline
from pykeen import predict
from pykeen.datasets import get_dataset


def main(args):
    print(f'Running evaluation on {args.dataset} for {args.model}')
    t0 = time.time()

    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)

    try:

        # load dataset
        dataset = get_dataset(dataset=args.dataset)

        ############## TRAINING ##############

        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)

        # tracking_mode = machine (default) or process
        # CPU: uses RAPL files (only for intel CPUs with root access)
        # pynvml library (only for Nvidia GPUs) measures consumption of the whole machine and not only the process

        emissions_tracker.start()
        start_time = time.time()

        result = pipeline(
            dataset=dataset,
            model=args.model,
            training_kwargs=dict(
                num_epochs=args.epochs,
                use_tqdm_batch=False
            ),
            stopper='early',
            random_seed=args.seed
        )

        end_time = time.time()
        emissions_tracker.stop()

        results = {
            'start': start_time,
            'end': end_time,
            'model': None
        }

        # write results
        with open(os.path.join(output_dir, f'results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

        setattr(args, 'train_logdir', output_dir)

        ############## INFERENCE ##############
        split = 'validation'
        output_dir = create_output_dir(args.output_dir, 'infer', args.__dict__)

        start_time = time.time()

        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)

        # PyKEEN Inference

        # choose a triple randomly to test inference
        random.seed(args.seed)
        training_triples = dataset.training.mapped_triples
        random_index = random.randint(0, len(training_triples))
        test_triple = training_triples[random_index]

        emissions_tracker.start()

        # Tail Prediction
        df = predict.predict_target(
            model=result.model,
            head=int(test_triple[0]),
            relation=int(test_triple[1]),
            triples_factory=result.training
        ).df

        df.sort_values(by=['score'])

        emissions_tracker.stop()

        end_time = time.time()

        model_stats = {
            'params': result.model.num_parameters,
            'fsize': result.model.num_parameter_bytes
        }

        results = {
            'metrics': {},
            'start': start_time,
            'end': end_time,
            'model': model_stats
        }

        # predictive quality metrics
        q_metrics = {
            'Hits@1': result.get_metric('hits@1'),
            'Hits@3': result.get_metric('hits@3'),
            'Hits@5': result.get_metric('hits@5'),
            'Hits@10': result.get_metric('hits@10'),
            'Mean Reciprocal Rank': result.get_metric('mean_reciprocal_rank')
        }
        results['metrics'] = q_metrics

        # write results
        with open(os.path.join(output_dir, f'{split}_results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

        ############## FINALIZE ##############

        print(
            f"Evaluation finished in {timedelta(seconds=int(time.time() - t0))} seconds, results can be found in {output_dir}\n")
        return output_dir

    except Exception as e:
        print('ERROR\n', e)
        with open(os.path.join(output_dir, f'error.txt'), 'a') as f:
            f.write(str(e))
            f.write('\n')
            f.write(traceback.format_exc())
        if "emissions_tracker" in locals().keys():
            emissions_tracker.stop()
        raise RuntimeError(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='countries')
    parser.add_argument('--model', default='TransE')
    parser.add_argument('--output-dir', default='logs/kgem')
    parser.add_argument('--epochs', type=int, default=100)

    # randomization and hardware profiling
    parser.add_argument("--cpu-monitor-interval", type=float, default=0.5,
                        help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    main(args)
