import argparse
import traceback
from datetime import timedelta
import json
import os
import time
import random
from datetime import datetime
from pathlib import Path

from codecarbon import OfflineEmissionsTracker

import methods
from mlprops.util import create_output_dir, PatchedJSONEncoder

from pykeen.pipeline import pipeline
from pykeen import predict
from pykeen.datasets import get_dataset
from pykeen.training import SLCWATrainingLoop
from pykeen.stoppers import EarlyStopper
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation import LCWAEvaluationLoop
import torch
from pykeen.constants import PYKEEN_CHECKPOINTS


def main(args):
    print(f'Running evaluation on {args.dataset} for {args.model}')
    t0 = time.time()

    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)

    try:

        # load dataset
        dataset = get_dataset(dataset=args.dataset)

        # define model
        model = methods.model_init(args.model, dataset, args.seed)

        ############## TRAINING ##############

        # save checkpoints to load model for inference
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        checkpoint = f'{args.model}_{args.dataset}_checkpoint_{timestamp}.pt'

        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
        emissions_tracker.start()
        start_time = time.time()

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=dataset.training
        )
        _ = training_loop.train(
            triples_factory=dataset.training,
            num_epochs=args.epochs,
            checkpoint_name=checkpoint,
            checkpoint_directory=r'\Users\borow\kgem\checkpoints',
            checkpoint_frequency=30,
        )
        # returns the losses per epoch (not the trained model)

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

        ############## Evaluation #############
        output_dir = create_output_dir(args.output_dir, 'evaluate', args.__dict__)
        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
        emissions_tracker.start()
        start_time = time.time()

        evaluator = RankBasedEvaluator()
        evaluation_loop = LCWAEvaluationLoop(
            model=model,
            triples_factory=dataset.testing
        )
        result = evaluation_loop.evaluate()

        end_time = time.time()
        emissions_tracker.stop()

        results = {
            'start': start_time,
            'end': end_time
        }

        with open(os.path.join(output_dir, f'results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

        ############## INFERENCE ##############
        split = 'validation'
        output_dir = create_output_dir(args.output_dir, 'infer', args.__dict__)

        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
        emissions_tracker.start()
        start_time = time.time()

        # PyKEEN Inference

        # load trained model from checkpoint
        cpath = Path(r'\Users\borow\kgem\checkpoints')
        checkpoint = torch.load(cpath.joinpath(checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])

        # choose a triple randomly to test inference
        random.seed(args.seed)
        training_triples = dataset.training.mapped_triples
        random_index = random.randint(0, len(training_triples))
        test_triple = training_triples[random_index]

        # tail prediction
        df = predict.predict_target(
            model=model,
            head=int(test_triple[0]),
            relation=int(test_triple[1]),
            triples_factory=model.training
        ).df

        df.sort_values(by=['score'])

        end_time = time.time()
        emissions_tracker.stop()

        model_stats = {
            'params': model.num_parameters,
            'fsize': model.num_parameter_bytes
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
    parser.add_argument('--output-dir', default=r'\Users\borow\kgem\ergebnisse')
    parser.add_argument('--epochs', type=int, default=100)

    # randomization and hardware profiling
    parser.add_argument("--cpu-monitor-interval", type=float, default=0.5,
                        help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    main(args)
