""" Utilities for viewing MRI images and interactively identify electrodes

Includes version of OrthoSlicer3D code originally written by:
Paul Ivanov, Eric Larson, Matthew Brett
"""
from __future__ import division, print_function

from time import time
import itertools
from pandas import DataFrame as df
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from .models import ModelSurface

import numpy as np
import weakref

from nibabel.optpkg import optional_package
from nibabel.orientations import aff2axcodes, axcodes2ornt


class OrthoSlicer3D(object):
    """Orthogonal-plane slicer.

    OrthoSlicer3d expects 3-dimensional data, and by default it creates a
    figure with 3 axes, one for each slice orientation.

    Clicking and dragging the mouse in any one axis will select out the
    corresponding slices in the other two. Scrolling up and
    down moves the slice up and down in the current axis.

    Example
    -------
    >>> import numpy as np
    >>> a = np.sin(np.linspace(0,np.pi,20))
    >>> b = np.sin(np.linspace(0,np.pi*5,20))
    >>> data = np.outer(a,b)[..., np.newaxis]*a
    >>> OrthoSlicer3D(data).show()  # doctest: +SKIP
    """
    # Skip doctest above b/c not all systems have mpl installed
    def __init__(self, data, affine=None, axes=None, cmap='gray',
                 pcnt_range=(1., 99.), figsize=(8, 8), title=None):
        """
        Parameters
        ----------
        data : ndarray
            The data that will be displayed by the slicer. Should have 3+
            dimensions.
        affine : array-like | None
            Affine transform for the data. This is used to determine
            how the data should be sliced for plotting into the saggital,
            coronal, and axial view axes. If None, identity is assumed.
            The aspect ratio of the data are inferred from the affine
            transform.
        axes : tuple of mpl.Axes | None, optional
            3 or 4 axes instances for the 3 slices plus volumes,
            or None (default).
        cmap : str | instance of cmap, optional
            String or cmap instance specifying colormap.
        pcnt_range : array-like, optional
            Percentile range over which to scale image for display.
        figsize : tuple
            Figure size (in inches) to use if axes are None.
        """
        # Nest imports so that matplotlib.use() has the appropriate
        # effect in testing
        plt, _, _ = optional_package('matplotlib.pyplot')
        mpl_img, _, _ = optional_package('matplotlib.image')
        mpl_patch, _, _ = optional_package('matplotlib.patches')
        self._title = title
        self._closed = False

        data = np.asanyarray(data)
        if data.ndim < 3:
            raise ValueError('data must have at least 3 dimensions')
        affine = np.array(affine, float) if affine is not None else np.eye(4)
        if affine.ndim != 2 or affine.shape != (4, 4):
            raise ValueError('affine must be a 4x4 matrix')
        # determine our orientation
        self._affine = affine.copy()
        codes = axcodes2ornt(aff2axcodes(self._affine))
        self._order = np.argsort([c[0] for c in codes])
        self._flips = np.array([c[1] < 0 for c in codes])[self._order]
        self._flips = list(self._flips) + [False]  # add volume dim
        self._scalers = np.abs(self._affine).max(axis=0)[:3]
        self._inv_affine = np.linalg.inv(affine)
        # current volume info
        self._volume_dims = data.shape[3:]
        self._current_vol_data = data[:, :, :, 0] if data.ndim > 3 else data
        self._data = data
        vmin, vmax = np.percentile(data, pcnt_range)
        del data

        if axes is None:  # make the axes
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |   Sag   |     |   Cor   |
            # S |    0    |   S |    1    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #        A  -->     <--  R
            # ^ +---------+     +---------+
            # | |         |     |         |
            #   |  Axial  |     |   Vol   |
            # A |    2    |     |    3    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #   <--  R          <--  t  -->

            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches(figsize, forward=True)
            self._axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
            plt.tight_layout(pad=0.1)
            if self.n_volumes <= 1:
                fig.delaxes(self._axes[3])
                self._axes.pop(-1)
            if self._title is not None:
                fig.canvas.set_window_title(str(title))
        else:
            self._axes = [axes[0], axes[1], axes[2]]
            if len(axes) > 3:
                self._axes.append(axes[3])

        # Start midway through each axis, idx is current slice number
        self._ims, self._data_idx = list(), list()

        # set up axis crosshairs
        self._crosshairs = [None] * 3
        r = [self._scalers[self._order[2]] / self._scalers[self._order[1]],
             self._scalers[self._order[2]] / self._scalers[self._order[0]],
             self._scalers[self._order[1]] / self._scalers[self._order[0]]]
        self._sizes = [self._data.shape[o] for o in self._order]
        for ii, xax, yax, ratio, label in zip([0, 1, 2], [1, 0, 0], [2, 2, 1],
                                              r, ('SAIP', 'SLIR', 'ALPR')):
            ax = self._axes[ii]
            d = np.zeros((self._sizes[yax], self._sizes[xax]))
            im = self._axes[ii].imshow(d, vmin=vmin, vmax=vmax, aspect=1,
                                       cmap=cmap, interpolation='nearest',
                                       origin='lower')
            self._ims.append(im)
            vert = ax.plot([0] * 2, [-0.5, self._sizes[yax] - 0.5],
                           color=(0, 1, 0), linestyle='-')[0]
            horiz = ax.plot([-0.5, self._sizes[xax] - 0.5], [0] * 2,
                            color=(0, 1, 0), linestyle='-')[0]
            self._crosshairs[ii] = dict(vert=vert, horiz=horiz)
            # add text labels (top, right, bottom, left)
            lims = [0, self._sizes[xax], 0, self._sizes[yax]]
            bump = 0.01
            poss = [[lims[1] / 2., lims[3]],
                    [(1 + bump) * lims[1], lims[3] / 2.],
                    [lims[1] / 2., 0],
                    [lims[0] - bump * lims[1], lims[3] / 2.]]
            anchors = [['center', 'bottom'], ['left', 'center'],
                       ['center', 'top'], ['right', 'center']]
            for pos, anchor, lab in zip(poss, anchors, label):
                ax.text(pos[0], pos[1], lab,
                        horizontalalignment=anchor[0],
                        verticalalignment=anchor[1])
            ax.axis(lims)
            ax.set_aspect(ratio)
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            self._data_idx.append(0)
        self._data_idx.append(-1)  # volume

        # Set up volumes axis
        if self.n_volumes > 1 and len(self._axes) > 3:
            ax = self._axes[3]
            ax.set_axis_bgcolor('k')
            ax.set_title('Volumes')
            y = np.zeros(self.n_volumes + 1)
            x = np.arange(self.n_volumes + 1) - 0.5
            step = ax.step(x, y, where='post', color='y')[0]
            ax.set_xticks(np.unique(np.linspace(0, self.n_volumes - 1,
                                                5).astype(int)))
            ax.set_xlim(x[0], x[-1])
            yl = [self._data.min(), self._data.max()]
            yl = [l + s * np.diff(lims)[0] for l, s in zip(yl, [-1.01, 1.01])]
            patch = mpl_patch.Rectangle([-0.5, yl[0]], 1., np.diff(yl)[0],
                                        fill=True, facecolor=(0, 1, 0),
                                        edgecolor=(0, 1, 0), alpha=0.25)
            ax.add_patch(patch)
            ax.set_ylim(yl)
            self._volume_ax_objs = dict(step=step, patch=patch)

        self._figs = set([a.figure for a in self._axes])
        for fig in self._figs:
            fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self._on_mouse)
            fig.canvas.mpl_connect('button_press_event', self._on_mouse)
            fig.canvas.mpl_connect('key_press_event', self._on_keypress)
            fig.canvas.mpl_connect('close_event', self._cleanup)

        # actually set data meaningfully
        self._position = np.zeros(4)
        self._position[3] = 1.  # convenience for affine multn
        self._changing = False  # keep track of status to avoid loops
        self._links = []  # other viewers this one is linked to
        plt.draw()
        for fig in self._figs:
            fig.canvas.draw()
        self._set_volume_index(0, update_slices=False)
        self._set_position(0., 0., 0.)
        self._draw()

    def __repr__(self):
        title = '' if self._title is None else ('%s ' % self._title)
        vol = '' if self.n_volumes <= 1 else (', %s' % self.n_volumes)
        r = ('<%s: %s(%s, %s, %s%s)>'
             % (self.__class__.__ch_idx__, title, self._sizes[0],
                self._sizes[1], self._sizes[2], vol))
        return r

    # User-level functions ###################################################
    def show(self):
        """Show the slicer in blocking mode; convenience for ``plt.show()``
        """
        plt, _, _ = optional_package('matplotlib.pyplot')
        plt.show()

    def close(self):
        """Close the viewer figures
        """
        self._cleanup()
        plt, _, _ = optional_package('matplotlib.pyplot')
        for f in self._figs:
            plt.close(f)

    def _cleanup(self):
        """Clean up before closing"""
        self._closed = True
        for link in list(self._links):  # make a copy before iterating
            self._unlink(link())

    @property
    def n_volumes(self):
        """Number of volumes in the data"""
        return int(np.prod(self._volume_dims))

    @property
    def position(self):
        """The current coordinates"""
        return self._position[:3].copy()

    def link_to(self, other):
        """Link positional changes between two canvases

        Parameters
        ----------
        other : instance of OrthoSlicer3D
            Other viewer to use to link movements.
        """
        if not isinstance(other, self.__class__):
            raise TypeError('other must be an instance of %s, not %s'
                            % (self.__class__.__ch_idx__, type(other)))
        self._link(other, is_primary=True)

    def _link(self, other, is_primary):
        """Link a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            return
        self._links.append(ref)
        if is_primary:
            other._link(self, is_primary=False)
            other.set_position(*self.position)

    def _unlink(self, other):
        """Unlink a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            self._links.pop(self._links.index(ref))
            ref()._unlink(self)

    def _notify_links(self):
        """Notify linked canvases of a position change"""
        for link in self._links:
            link().set_position(*self.position[:3])

    def set_position(self, x=None, y=None, z=None):
        """Set current displayed slice indices

        Parameters
        ----------
        x : float | None
            X coordinate to use. If None, do not change.
        y : float | None
            Y coordinate to use. If None, do not change.
        z : float | None
            Z coordinate to use. If None, do not change.
        """
        self._set_position(x, y, z)
        self._draw()

    def set_volume_idx(self, v):
        """Set current displayed volume index

        Parameters
        ----------
        v : int
            Volume index.
        """
        self._set_volume_index(v)
        self._draw()

    def _set_volume_index(self, v, update_slices=True):
        """Set the plot data using a volume index"""
        v = self._data_idx[3] if v is None else int(round(v))
        if v == self._data_idx[3]:
            return
        max_ = np.prod(self._volume_dims)
        self._data_idx[3] = max(min(int(round(v)), max_ - 1), 0)
        idx = (slice(None), slice(None), slice(None))
        if self._data.ndim > 3:
            idx = idx + tuple(np.unravel_index(self._data_idx[3],
                                               self._volume_dims))
        self._current_vol_data = self._data[idx]
        # update all of our slice plots
        if update_slices:
            self._set_position(None, None, None, notify=False)

    def _set_position(self, x, y, z, notify=True):
        """Set the plot data using a physical position"""
        # deal with volume first
        if self._changing:
            return
        self._changing = True
        x = self._position[0] if x is None else float(x)
        y = self._position[1] if y is None else float(y)
        z = self._position[2] if z is None else float(z)

        # deal with slicing appropriately
        self._position[:3] = [x, y, z]
        idxs = np.dot(self._inv_affine, self._position)[:3]
        for ii, (size, idx) in enumerate(zip(self._sizes, idxs)):
            self._data_idx[ii] = max(min(int(round(idx)), size - 1), 0)
        for ii in range(3):
            # saggital: get to S/A
            # coronal: get to S/L
            # axial: get to A/L
            data = np.take(self._current_vol_data, self._data_idx[ii],
                           axis=self._order[ii])
            xax = [1, 0, 0][ii]
            yax = [2, 2, 1][ii]
            if self._order[xax] < self._order[yax]:
                data = data.T
            if self._flips[xax]:
                data = data[:, ::-1]
            if self._flips[yax]:
                data = data[::-1]
            self._ims[ii].set_data(data)
            # deal with crosshairs
            loc = self._data_idx[ii]
            if self._flips[ii]:
                loc = self._sizes[ii] - loc
            loc = [loc] * 2
            if ii == 0:
                self._crosshairs[2]['vert'].set_xdata(loc)
                self._crosshairs[1]['vert'].set_xdata(loc)
            elif ii == 1:
                self._crosshairs[2]['horiz'].set_ydata(loc)
                self._crosshairs[0]['vert'].set_xdata(loc)
            else:  # ii == 2
                self._crosshairs[1]['horiz'].set_ydata(loc)
                self._crosshairs[0]['horiz'].set_ydata(loc)

        # Update volume trace
        if self.n_volumes > 1 and len(self._axes) > 3:
            idx = [None, Ellipsis] * 3
            for ii in range(3):
                idx[self._order[ii]] = self._data_idx[ii]
            vdata = self._data[idx].ravel()
            vdata = np.concatenate((vdata, [vdata[-1]]))
            self._volume_ax_objs['patch'].set_x(self._data_idx[3] - 0.5)
            self._volume_ax_objs['step'].set_ydata(vdata)
        if notify:
            self._notify_links()
        self._changing = False

    # Matplotlib handlers ####################################################
    def _in_axis(self, event):
        """Return axis index if within one of our axes, else None"""
        if getattr(event, 'inaxes') is None:
            return None
        for ii, ax in enumerate(self._axes):
            if event.inaxes is ax:
                return ii

    def _on_scroll(self, event):
        """Handle mpl scroll wheel event"""
        assert event.button in ('up', 'down')
        ii = self._in_axis(event)
        if ii is None:
            return
        if event.key is not None and 'shift' in event.key:
            if self.n_volumes <= 1:
                return
            ii = 3  # shift: change volume in any axis
        assert ii in range(4)
        dv = 10. if event.key is not None and 'control' in event.key else 1.
        dv *= 1. if event.button == 'up' else -1.
        dv *= -1 if self._flips[ii] else 1
        val = self._data_idx[ii] + dv
        if ii == 3:
            self._set_volume_index(val)
        else:
            coords = [self._data_idx[k] for k in range(3)] + [1.]
            coords[ii] = val
            self._set_position(*np.dot(self._affine, coords)[:3])
        self._draw()

    def _on_mouse(self, event):
        """Handle mpl mouse move and button press events"""
        if event.button != 1:  # only enabled while dragging
            return
        ii = self._in_axis(event)
        if ii is None:
            return
        if ii == 3:
            # volume plot directly translates
            self._set_volume_index(event.xdata)
        else:
            # translate click xdata/ydata to physical position
            xax, yax = [[1, 2], [0, 2], [0, 1]][ii]
            x, y = event.xdata, event.ydata
            x = self._sizes[xax] - x if self._flips[xax] else x
            y = self._sizes[yax] - y if self._flips[yax] else y
            idxs = [None, None, None, 1.]
            idxs[xax] = x
            idxs[yax] = y
            idxs[ii] = self._data_idx[ii]
            self._set_position(*np.dot(self._affine, idxs)[:3])
        self._draw()

    def _on_keypress(self, event):
        """Handle mpl keypress events"""
        if event.key is not None and 'escape' in event.key:
            self.close()

    def _draw(self):
        """Update all four (or three) plots"""
        if self._closed:  # make sure we don't draw when we shouldn't
            return
        for ii in range(3):
            ax = self._axes[ii]
            ax.draw_artist(self._ims[ii])
            for line in self._crosshairs[ii].values():
                ax.draw_artist(line)
            ax.figure.canvas.blit(ax.bbox)
        if self.n_volumes > 1 and len(self._axes) > 3:
            ax = self._axes[3]
            ax.draw_artist(ax.patch)  # axis bgcolor to erase old lines
            for key in ('step', 'patch'):
                ax.draw_artist(self._volume_ax_objs[key])
            ax.figure.canvas.blit(ax.bbox)


class ElectrodeGUI():
    """Graphical user interface to identify electrodes locations.

    Parameters
    ==========
    niimg : str | nilearn image
        The MRI object or filename.
    xy : np.array, shape(n_channels, 3)
        Channels known 2D locations.
    alpha : float, in [0, 1]
        Surface fitting rigidity constrain, see `ModelSurface`.
    fit : bool,
        Apply fitting
    cmap : str | mpl.cmap
        MRI cmap
    """
    def __init__(self, niimg, xy=None, alpha=.5, ch_user=None, fit=True,
                 cmap='magma'):
        from nilearn import _utils
        if isinstance(niimg, str):
            niimg = _utils(niimg)
        # Store params
        self.fit = fit
        self.alpha = alpha
        self.xy = xy

        # Read MRI data
        data = niimg.get_data()
        affine = niimg.affine

        # Setup MRI viewer
        self.axes = list()
        # ---- Horizontally stack the three views
        widths = np.array(np.abs([affine[1, -1], affine[0, -1],
                                  affine[0, -1]]), int)
        cum_width = 0
        for ii, width in enumerate(widths):
            ax = plt.subplot2grid((1, sum(widths)), (0, cum_width),
                                  colspan=width)
            self.axes.append(ax)
            cum_width += width
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1.0, top=1.,
                            bottom=0.1)
        self.viewer = OrthoSlicer3D(data, affine=affine, cmap=cmap,
                                    axes=self.axes)
        # ----  Each view has a particular x, y, depth order
        self.ax_xyd = ([1, 2, 0], [0, 2, 1], [0, 1, 2])

        # Interactive components
        ax.get_figure().canvas.mpl_connect('key_press_event', self._press)

        # channel locations
        columns = (['handle_%i' % ii for ii in range(3)] +
                   ['x', 'y', 'z', 'x_idx', 'y_idx', 'z_idx'])
        self.ch_user = df(columns=columns) if ch_user is None else ch_user
        self.ch_pred = df(columns=columns)

        # Scatter plot handles
        plt.gcf().canvas.mpl_connect('motion_notify_event', self._draw)
        self._last_refresh = time()  # XXX to refresh every 100 ms max

        # Grid GUI
        self.grid = GridGUI(ax=plt.axes([0.1, 0., 0.1, 0.1]), xy=xy)

        # Setup surface fitting
        self.model = ModelSurface(alpha=alpha)

        self._init_add()
        plt.show()

    def _draw(self, event=None):
        """Refresh known and predicted channels colors and sizes.
        """
        # XXX refresh on every slide move: probably not the best approach..
        if (time() + .100) < self._last_refresh:
            return
        if not len(self.ch_user):
            return
        # Current view:
        x, y, z, _ = self._current_idx()
        # Update each view
        for ax, (current_depth, d_dim) in enumerate(zip(
                [x, y, z], ['x_idx', 'y_idx', 'z_idx'])):
            # from channel image index coord. to axis x, y depth coord.
            max_depth = self.viewer._data.shape[ax]

            # Update true and predicted channels
            for ch, typ in zip([self.ch_user, self.ch_pred], ['true', 'pred']):
                handles = ch['handle_%i' % ax]
                depths = ch[d_dim]

                # Depth relative to current xyz normalize by img size
                relative_depths = (depths - current_depth) / float(max_depth)

                # Change color and size
                colors = np.array(['w', 'r', 'b'])
                colors = colors[np.array(np.sign(relative_depths), int)]
                sizes = np.exp((1. - np.abs(relative_depths)) * 5) / 10

                # Update
                for color, size, ha in zip(colors, sizes, handles):
                    ha.set_color(color)
                    ha.set_sizes([size])
                    if typ == 'pred':
                        ha.set_facecolor('none')

            plt.draw()
        self._last_refresh = time()

    def _press(self, event):
        import sys
        sys.stdout.flush()
        if event.key == u' ':
            self._add_here()
        elif event.key == u'delete':
            self._remove()  # TODO
        elif event.key == u'h':
            self._hide()  # TODO
        else:
            return
        self._draw()

    def _remove(self, event):
        raise NotImplementedError()

    def _hide(self, event):
        raise NotImplementedError()

    def _init_add(self):
        """
        Plot all known electrodes
        """
        # TODO check ch_user, add ch_idx if not provided
        for loc in range(len(self.ch_user)):
            # Plot new channel on each axis
            self._add(*self.ch_user[['x', 'y', 'z', 'ch_idx']])

    def _add_here(self):
        ch_idx = self.grid.selected_channel_  # current channel
        # Ensure that this point hasn't already been defined
        if 'ch_idx' in self.ch_user.keys():
            iloc = self.ch_user['ch_idx'] == ch_idx
            if sum(iloc):
                # remove dots
                for ax in range(3):
                    self.ch_user['handle_%i' % ax][iloc].values[0].remove()
                # remove line
                self.ch_user = self.ch_user[~iloc]

        # Retrieve channel information
        x, y, z, _ = self.viewer._position  # current position
        # Plot and store new channel
        channel = self._add(x, y, z, ch_idx)
        self.ch_user = self.ch_user.append(channel, ignore_index=True)

        # Fit and predict channel locations
        if len(self.ch_user) > 2 and self.fit:
            # Clean previous dots
            n_chan = len(self.ch_pred)
            for ch, ax in itertools.product(range(n_chan), range(3)):
                self.ch_pred['handle_%i' % ax][ch].remove()
            self.ch_pred = self.ch_pred[[False] * n_chan]  # clear
            # Get 3D coordinates of identified channels
            y = self.ch_user[['x', 'y', 'z']].values
            idx = np.array(self.ch_user['ch_idx'].values, int)
            self.model.fit(X=self.grid.xy, y=y, idx=idx)
            # Predict all channels
            xyz = self.model.predict(self.grid.xy)
            # Plot and store predicted channels
            for (x, y, z), ch_idx in zip(xyz, idx):
                channel = self._add(x, y, z, ch_idx)
                self.ch_pred = self.ch_pred.append(channel, ignore_index=True)

        self._draw()

    def _add(self, x, y, z, ch_idx):
        xyz_idx = self._inv_affine([x, y, z, 1])[0]
        x_2D, y_2D = self.grid.xy[ch_idx, :].tolist()
        channel = dict(x=x, y=y, z=z, x_2D=x_2D, y_2D=y_2D, ch_idx=ch_idx,
                       x_idx=xyz_idx[0], y_idx=xyz_idx[1], z_idx=xyz_idx[2])

        # Plot new channel on each axis
        for ax, order in enumerate(self.ax_xyd):
            # Transform 3D coordinates into 2D + depths for each axis
            x_dim, y_dim, _ = np.array(['x_idx', 'y_idx', 'z_idx'])[order]
            x = channel[x_dim]
            y = channel[y_dim]
            # Add scatter
            channel['handle_%i' % ax] = self.axes[ax].scatter(x, y)
        return channel

    def _current_idx(self):
        """MRI data indices of current view"""
        return self._inv_affine(self.viewer._position)[0]

    def _inv_affine(self, xyz):
        """From head coordinates to image indices"""
        xyz = np.reshape(xyz, [-1, 4])  # ensure array for multiple positions
        # Deal with left / right flips:
        flips = np.array([1, -1])[self.viewer._flips]
        flips = np.tile(flips, [len(xyz), 1])
        # from head coordinates to image coordinate
        xyz_idx = np.dot(self.viewer._inv_affine, np.transpose(flips * xyz)).T
        return np.array(xyz_idx, int)


class GridGUI():
    """Graphical interface to see the channels in 2D, and select the one
    that needs to be identified.

    Parameters
    ==========
        ax : mpl ax
            Matplotlib axis instance.
        xy : np.array, shape(n_channels, 2)
            2D coordinates of the channels, in mm. Defaults to grid of 4 x 4
            channels regularly separated by 10 mm.
    Attributes:
        selected_channel_ : int
            Index of the currently selected channel.
    """
    def __init__(self, ax=None, xy=None):
        # Set default axis
        if ax is None:
            ax = plt.subplots(1)[1]
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax = ax

        # Default grid of channels
        if xy is None:
            xy = np.meshgrid(np.linspace(0, 30., 4), np.linspace(0, 30., 4))
            xy = np.transpose([ii.ravel() for ii in xy])
        self.xy = np.array(xy)
        self.selected_channel_ = 0  # currently selected channel

        # plot channels
        self._scatter_handle = self.ax.scatter(self.xy[:, 0],
                                               self.xy[:, 1], 20, 'k')
        self._update_color()
        plt.gcf().canvas.mpl_connect('button_press_event', self._select)

    def _select(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        ind = self._get_ind_under_point(event.xdata, event.ydata)
        if ind is None:  # mouse press too far from dot
            return
        self.selected_channel_ = ind
        self._update_color()

    def _update_color(self):
        colors = np.zeros((len(self.xy), 3))
        colors[self.selected_channel_] = [1., 0., 0.]
        self._scatter_handle.set_color(colors)
        plt.draw()

    def _get_ind_under_point(self, x, y):
        """Finds which dots the mouse press is on."""
        distance = np.sqrt((self.xy[:, 0] - x) ** 2 +
                           (self.xy[:, 1] - y) ** 2)
        ind = np.argmin(distance)
        epsilon = 5  # min distance in pixel
        if distance[ind] >= epsilon:
            ind = None
        return ind
