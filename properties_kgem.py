
PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['dataset'],
        'model': lambda log: log['config']['model'],
        'architecture': lambda log: log['execution_platform']['Processor'],
        'software': lambda log: 'PyKEEN 1.10.1',
    },

    'train': {
        'train_running_time': lambda log: log['emissions']['duration']['0'],
        'train_power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6
    },

    'infer': {
        'running_time': lambda log: log['emissions']['duration']['0'],
        'power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6,
        'Hits@5': lambda log: log['validation_results']['metrics']['Hits@5'],
        'Hits@10': lambda log: log['validation_results']['metrics']['Hits@10'],
        'Mean Reciprocal Rank': lambda log: log['validation_results']['metrics']['Mean Reciprocal Rank'],
        'parameters': lambda log: log['validation_results']['model']['params'],
        'fsize': lambda log: log['validation_results']['model']['fsize'],
    }
}
