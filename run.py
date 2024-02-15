import argparse
import traceback
from datetime import timedelta
import json
import os
import time
import random
import logging

from codecarbon import OfflineEmissionsTracker
# from pykeen.evaluation import RankBasedEvaluator

from strep.util import create_output_dir, PatchedJSONEncoder

from pykeen import predict
from pykeen.datasets import get_dataset
from pykeen.stoppers import stopper_resolver, Stopper
from pykeen.trackers import resolve_result_trackers
from pykeen.pipeline.api import _handle_random_seed, _handle_dataset, _handle_model, _handle_training_loop, \
    _handle_evaluator, _handle_evaluation
from pykeen.utils import set_random_seed
from pykeen.pipeline import PipelineResult


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

        # evaluator = RankBasedEvaluator(batch_size=512)

        ### Taken from pykeen.pipeline.api

        # pipeline configuration
        # 1. Dataset
        dataset = dataset
        dataset_kwargs = dict(
            create_inverse_triples=False,
        )
        training = None
        testing = None
        validation = None
        evaluation_entity_whitelist = None
        evaluation_relation_whitelist = None
        # 2. Model
        model = args.model
        model_kwargs = None
        interaction = None
        interaction_kwargs = None
        dimensions = None
        # 3. Loss
        loss = 'marginrankingloss'
        loss_kwargs = None
        # 4. Regularizer
        regularizer = None
        regularizer_kwargs = None
        # 5. Optimizer
        optimizer = None
        optimizer_kwargs = None
        clear_optimizer = True
        # 5.1 Learning Rate Scheduler
        lr_scheduler = None
        lr_scheduler_kwargs = None
        # 6. Training Loop
        training_loop = 'sLCWA'
        training_loop_kwargs = None
        negative_sampler = None
        negative_sampler_kwargs = None
        # 7. Training (ronaldo style)
        epochs = None
        training_kwargs = dict(
            num_epochs=args.epochs,
            use_tqdm_batch=False
        )
        stopper = 'early'
        stopper_kwargs = None
        # 8. Evaluation
        evaluator = None
        # evaluator = evaluator
        evaluator_kwargs = None
        evaluation_kwargs = None
        # evaluation_kwargs = dict(batch_size=512)
        # 9. Tracking
        result_tracker = None
        result_tracker_kwargs = None
        # Misc
        metadata = None
        device = None
        random_seed = args.seed
        use_testing_data = True
        evaluation_fallback = False
        filter_validation_when_testing = True
        use_tqdm = None

        if training_kwargs is None:
            training_kwargs = {}
        training_kwargs = dict(training_kwargs)

        _random_seed, clear_optimizer = _handle_random_seed(
            training_kwargs=training_kwargs, random_seed=random_seed, clear_optimizer=clear_optimizer
        )
        set_random_seed(_random_seed)

        _result_tracker = resolve_result_trackers(result_tracker, result_tracker_kwargs)

        if not metadata:
            metadata = {}
        title = metadata.get("title")

        # Start tracking
        _result_tracker.start_run(run_name=title)

        training, testing, validation = _handle_dataset(
            _result_tracker=_result_tracker,
            dataset=dataset,
            dataset_kwargs=dataset_kwargs,
            training=training,
            testing=testing,
            validation=validation,
            evaluation_entity_whitelist=evaluation_entity_whitelist,
            evaluation_relation_whitelist=evaluation_relation_whitelist,
        )

        model_instance = _handle_model(
            device=device,
            _result_tracker=_result_tracker,
            _random_seed=_random_seed,
            training=training,
            model=model,
            model_kwargs=model_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            dimensions=dimensions,
            loss=loss,
            loss_kwargs=loss_kwargs,
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
        )

        training_loop_instance = _handle_training_loop(
            _result_tracker=_result_tracker,
            model_instance=model_instance,
            training=training,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            training_loop=training_loop,
            training_loop_kwargs=training_loop_kwargs,
            negative_sampler=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
        )

        evaluator_instance, evaluation_kwargs = _handle_evaluator(
            _result_tracker=_result_tracker,
            evaluator=evaluator,
            evaluator_kwargs=evaluator_kwargs,
            evaluation_kwargs=evaluation_kwargs,
        )

        # _handle_training
        _result_tracker = _result_tracker
        training = training
        validation = validation
        model_instance = model_instance
        evaluator_instance = evaluator_instance
        training_loop_instance = training_loop_instance
        clear_optimizer = clear_optimizer
        evaluation_kwargs = evaluation_kwargs
        epochs = epochs
        training_kwargs = training_kwargs
        stopper = stopper
        stopper_kwargs = stopper_kwargs
        use_tqdm = use_tqdm

        # Stopping
        if "stopper" in training_kwargs and stopper is not None:
            raise ValueError("Specified stopper in training_kwargs and as stopper")
        if "stopper" in training_kwargs:
            stopper = training_kwargs.pop("stopper")
        if stopper_kwargs is None:
            stopper_kwargs = {}
        stopper_kwargs = dict(stopper_kwargs)

        # Load the evaluation batch size for the stopper, if it has been set
        _evaluation_batch_size = evaluation_kwargs.get("batch_size")
        if _evaluation_batch_size is not None:
            stopper_kwargs.setdefault("evaluation_batch_size", _evaluation_batch_size)

        stopper_instance: Stopper = stopper_resolver.make(
            stopper,
            model=model_instance,
            evaluator=evaluator_instance,
            training_triples_factory=training,
            evaluation_triples_factory=validation,
            result_tracker=_result_tracker,
            **stopper_kwargs,
        )

        if epochs is not None:
            training_kwargs["num_epochs"] = epochs
        if use_tqdm is not None:
            training_kwargs["use_tqdm"] = use_tqdm
        training_kwargs.setdefault("num_epochs", 5)
        training_kwargs.setdefault("batch_size", 256)
        _result_tracker.log_params(params=training_kwargs)

        # Add logging for debugging
        configuration = _result_tracker.get_configuration()
        logging.debug("Run Pipeline based on following config:")
        for key, value in configuration.items():
            logging.debug(f"{key}: {value}")

        ########### Monitoring Training Function ################

        emissions_tracker.start()
        start_time = time.time()

        # Train like Cristiano Ronaldo
        training_start_time = time.time()
        losses = training_loop_instance.train(
            triples_factory=training,
            stopper=stopper_instance,
            clear_optimizer=clear_optimizer,
            **training_kwargs,
        )

        end_time = time.time()
        emissions_tracker.stop()

        ##########################################################
        assert losses is not None  # losses is only none if it's doing search mode
        training_end_time = time.time() - training_start_time
        step = training_kwargs.get("num_epochs")
        _result_tracker.log_metrics(metrics=dict(total_training=training_end_time), step=step, prefix="times")
        train_seconds = training_end_time

        metric_results, evaluate_seconds = _handle_evaluation(
            _result_tracker=_result_tracker,
            model_instance=model_instance,
            evaluator_instance=evaluator_instance,
            stopper_instance=stopper_instance,
            training=training,
            testing=testing,
            validation=validation,
            training_kwargs=training_kwargs,
            evaluation_kwargs=evaluation_kwargs,
            use_testing_data=use_testing_data,
            evaluation_fallback=evaluation_fallback,
            filter_validation_when_testing=filter_validation_when_testing,
            use_tqdm=use_tqdm,
        )
        _result_tracker.end_run()

        result = PipelineResult(
            random_seed=_random_seed,
            model=model_instance,
            training=training,
            training_loop=training_loop_instance,
            losses=losses,
            stopper=stopper_instance,
            configuration=configuration,
            metric_results=metric_results,
            metadata=metadata,
            train_seconds=train_seconds,
            evaluate_seconds=evaluate_seconds,
        )
        ### End of the Code from pykeen.pipeline.api

        results = {
            'history': {
                'loss': result.losses
            },
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

        emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.cpu_monitor_interval, log_level='warning',
                                                    country_iso_code="DEU", save_to_file=True, output_dir=output_dir)

        # choose a triple randomly for inference
        random.seed(args.seed)
        testing_triples = dataset.testing.mapped_triples
        random_index = random.randint(0, len(testing_triples))
        test_triple = testing_triples[random_index]

        emissions_tracker.start()
        start_time = time.time()

        # tail prediction
        df = predict.predict_target(
            model=result.model,
            head=int(test_triple[0]),
            relation=int(test_triple[1])
        ).df

        df.sort_values(by=['score'])

        end_time = time.time()
        emissions_tracker.stop()

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
    parser.add_argument('--dataset', default='kinships')
    parser.add_argument('--model', default='TransE')
    parser.add_argument('--output-dir', default='logs/kgem')
    parser.add_argument('--epochs', type=int, default=1000)

    # randomization and hardware profiling
    parser.add_argument("--cpu-monitor-interval", type=float, default=0.5,
                        help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    main(args)
