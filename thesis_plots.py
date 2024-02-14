import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import AxesGrid

from main import preprocess_database
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background
from strep.elex.util import RATING_COLORS, RATING_COLOR_SCALE_REV
from strep.load_experiment_logs import find_sub_db
from strep.util import lookup_meta, prop_dict_to_val, identify_all_correlations

PLOT_WIDTH = 1000
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b',
          '#ffffff']
CORR_COLORS = ['#e52421', '#ef7c29', '#f8b930', '#b8ac2b', '#649b30']

SEL_DS_TASK = [
    ('FB15k-237', 'infer'),
    ('WN18-RR', 'infer'),
    ('kinships', 'infer'),
    ('FB15k-237', 'train'),
    ('WN18-RR', 'train'),
    ('kinships', 'train')
]


def create_all(database):
    os.chdir('thesis_plots')

    rdb, meta, metrics, xdef, ydef, bounds, _, _ = preprocess_database(database)

    corr_res = []

    for ds_task in SEL_DS_TASK:

        # scatter plot
        ds, task = ds_task
        ds_name = lookup_meta(meta, ds, subdict='dataset')
        xaxis, yaxis = xdef[(ds, task)], ydef[(ds, task)]
        db = find_sub_db(rdb, dataset=ds, task=task)
        plot_data, axis_names, rating_pos = assemble_scatter_data([db['environment'].iloc[0]], db, 'index', xaxis,
                                                                  yaxis, meta, bounds)
        scatter = create_scatter_graph(plot_data, axis_names, dark_mode=False)
        rating_pos[0][0][0] = scatter.layout.xaxis.range[1]
        rating_pos[1][0][0] = scatter.layout.yaxis.range[1]
        add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False)
        scatter.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25},
                              title_y=0.99, title_x=0.5, title_text=f'{ds_name}')
        scatter.write_image(f"scatter_{ds}_{task}.pdf")

        # star plot
        db = prop_dict_to_val(db, 'index')
        worst = db.sort_values('compound_index').iloc[0]
        best = db.sort_values('compound_index').iloc[-1]
        fig = go.Figure()
        for model, col, m_str in zip([best, worst], [RATING_COLORS[0], RATING_COLORS[4]], ['Best', 'Worst']):
            mod_name = lookup_meta(meta, model['model'], 'short', 'model')[:18]
            metr_names = [lookup_meta(meta, metr, 'shortname', 'properties') for metr in metrics[(ds, task)]]
            fig.add_trace(go.Scatterpolar(
                r=[model[col] for col in metrics[(ds, task)]], line={'color': col},
                theta=metr_names, fill='toself', name=f'{mod_name} ({m_str}): {model["compound_index"]:4.2f}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), width=(PLOT_WIDTH + 20) * 0.25, height=PLOT_HEIGHT, title_y=0.95,
            title_x=0.5, title_text=ds_name,
            legend=dict(yanchor="bottom", y=1.06, xanchor="center", x=0.5), margin={'l': 50, 'r': 50, 'b': 15, 't': 90}
        )
        fig.write_image(f'true_best_{ds}_{task}.pdf')

        # COMPUTE CORRELATIONS
        correlations = {scale: identify_all_correlations(db, metrics, scale) for scale in ['index', 'value']}

        corr, props = correlations['index'][ds_task]
        prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in props]
        corr_res.append((corr, prop_names, ds_task))

        fig = go.Figure(data=go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"))

        fig.update_layout(coloraxis={'colorscale': RATING_COLOR_SCALE_REV, 'colorbar': {'title': 'Pearson Corr'}})
        fig.update_layout(
            {'width': PLOT_HEIGHT + 120, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 20}})

        # fig.show()
        fig.write_image(f'correlation_kgem_{ds}_{task}.pdf')

    #################################################
    # all correlations in one plot
    fig = plt.figure(figsize=(6, 4))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.30,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    count = 0
    for ax in grid:
        # ax.set_axis_off()
        corr, prop_names, ds_task = corr_res[count]
        im = ax.imshow(
            corr,
            LinearSegmentedColormap.from_list("mycmap", CORR_COLORS),
            # ListedColormap(CORR_COLORS),
            vmin=-1,
            vmax=1
        )
        ax.set_xticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        ax.set_yticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title(f"{ds_task[0]} {ds_task[1]}", fontsize=10)
        count = count + 1

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(-1, 1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])
    # plt.show()
    plt.savefig(f'correlations_kgem.pdf')

    # correlations infer
    fig = plt.figure(figsize=(6, 3))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.30,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    count = 0
    for ax in grid:
        # ax.set_axis_off()
        corr, prop_names, ds_task = corr_res[count]
        im = ax.imshow(
            corr,
            LinearSegmentedColormap.from_list("mycmap", CORR_COLORS),
            # ListedColormap(CORR_COLORS),
            vmin=-1,
            vmax=1
        )
        ax.set_xticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        ax.set_yticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title(f"{ds_task[0]} {ds_task[1]}", fontsize=10)
        count = count + 1

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(-1, 1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])
    # plt.show()
    plt.savefig(f'correlations_infer.pdf')

    # correlations train
    fig = plt.figure(figsize=(6, 3))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.30,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    count = 3
    for ax in grid:
        # ax.set_axis_off()
        corr, prop_names, ds_task = corr_res[count]
        im = ax.imshow(
            corr,
            LinearSegmentedColormap.from_list("mycmap", CORR_COLORS),
            # ListedColormap(CORR_COLORS),
            vmin=-1,
            vmax=1
        )
        ax.set_xticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        ax.set_yticks(np.arange(len(prop_names)), labels=prop_names, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title(f"{ds_task[0]} {ds_task[1]}", fontsize=10)
        count = count + 1

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(-1, 1, 0.5))
    # plt.show()
    plt.savefig(f'correlations_train.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--database", default=r'kgem-energy-efficiency\databases\kgem\database.pkl')

    args = parser.parse_args()

    data = args.database

    create_all(data)
