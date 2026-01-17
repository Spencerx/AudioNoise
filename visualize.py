import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, RectangleSelector, RadioButtons

import os
import argparse

# --- Constants ---
INITIAL_WINDOW_SEC = 2.0
BYTES_PER_SAMPLE = 4
# MAX_WIDTH_SEC removed, utilizing self.duration_sec instead
MAX_PLOT_POINTS = 5000   # Maximum points to plot per line

class WaveformVisualizer:
    def __init__(self, filenames, rate, min_zoom_samples=100):
        self.rate = rate
        self.navigating = False
        self.filenames = filenames
        self.min_zoom_samples = min_zoom_samples
        self.min_width_sec = min_zoom_samples / rate

        self.mapped_files = []
        self.lines = []
        self.max_samples = 0

        # Load files
        for f in filenames:
            try:
                fsize = os.path.getsize(f)
                samples = fsize // BYTES_PER_SAMPLE
                mm = np.memmap(f, dtype=np.int32, mode='r', shape=(samples,))
                self.mapped_files.append((mm, os.path.basename(f)))
                self.max_samples = max(self.max_samples, samples)
            except Exception as e:
                print(f"Error opening {f}: {e}")

        if not self.mapped_files:
            return

        self.duration_sec = self.max_samples / self.rate

        # Pre-allocate X-axis buffer to avoid allocations during plot updates
        self.t_buffer = np.zeros(MAX_PLOT_POINTS, dtype=np.float64)
        # Pre-allocate index buffer 0..N-1
        self.index_buffer = np.arange(MAX_PLOT_POINTS, dtype=np.float64)

        self.setup_ui()

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        # Manual "tight layout" to maximize space but keep room for slider
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)

        # Create line objects
        for _, name in self.mapped_files:
            line, = self.ax.plot([], [], linewidth=0.8, label=name)
            self.lines.append(line)

        self.ax.grid(True, which='both', linestyle=':', alpha=0.5)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend(loc='upper right', fontsize='x-small')

        # Use FuncFormatter to show float scales while keeping data as int32
        self.y_mode = 'Volt'
        def y_fmt(x, pos):
            if self.y_mode == 'Raw':
                val = np.uint32(x)
                return f"{val:09_X}"
            # Do "Engineering mode" by hand
            x = x / 2147483648
            suffix, milli = "", "ᴇ-3"
            if self.y_mode == 'Volt':
                # Assuming 1Vrms full-range signal
                x = x * 1.4142
                suffix, milli = "V", "mV"
            if x == 0:
                return "0"+suffix
            if abs(x) < 0.1:
                x = x*1000
                suffix = milli
            return f"{x:.2f}"+suffix

        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

        # Slider setup
        ax_slider = plt.axes([0.15, 0.05, 0.60, 0.03]) # Shrunk slightly to fit radio buttons
        self.slider = Slider(ax_slider, 'Time', 0, self.duration_sec, valinit=0)
        self.slider.on_changed(self.update_slider)

        # RadioButtons for Y-axis Mode
        ax_radio = plt.axes([0.80, 0.05, 0.15, 0.10])
        self.radio = RadioButtons(ax_radio, ('Raw', 'Scaled', 'Volt'), active=2)

        def set_y_mode(label):
            self.y_mode = label
            self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

            # Update labels
            if label == 'Raw':
                 self.ax.set_ylabel("Amplitude (Raw)")
            elif label == 'Scaled':
                 self.ax.set_ylabel("Amplitude (Normalized)")
            elif label == 'Volt':
                 self.ax.set_ylabel("Amplitude (Volts)")

            self.fig.canvas.draw_idle()

        self.radio.on_clicked(set_y_mode)

        # Scroll Zoom setup
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)

        # Custom Rectangle Selector
        self.rs = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False
        )

        # Initial View
        self.update_view(0, INITIAL_WINDOW_SEC)

        plt.show()

    def get_chunk(self, start_time, window_duration):
        start_sample = int(start_time * self.rate)
        # Ensure we don't request negative start
        start_sample = max(0, start_sample)

        end_sample = int((start_time + window_duration) * self.rate)

        # Determine downsampling step to keep plot fast
        total_samples = end_sample - start_sample
        step = 1
        if total_samples > MAX_PLOT_POINTS:
            step = int(np.ceil(total_samples / MAX_PLOT_POINTS))

        global_min_y, global_max_y = 2147483647, -2147483648
        has_data = False

        for line, (mm, _) in zip(self.lines, self.mapped_files):
            if start_sample >= mm.size:
                line.set_data([], [])
                continue

            # Safe end for this specific file
            safe_end = min(end_sample, mm.size)
            if safe_end <= start_sample:
                 line.set_data([], [])
                 continue

            # Strided slice (View into memory map - very fast)
            chunk = mm[start_sample:safe_end:step]

            if chunk.size > 0:
                # Generate Time Axis without allocation using pre-allocated buffer
                # We use the shared self.t_buffer

                # Careful: chunk.size might be <= MAX_PLOT_POINTS.
                # If chunk.size > MAX_PLOT_POINTS (edge case), we clamp or just allocate.
                # With calculated step, chunk.size should be <= MAX_PLOT_POINTS generally.

                current_count = chunk.size
                if current_count > len(self.t_buffer):
                    # Should rarely happen with correct step logic, but resize if needed
                    self.t_buffer = np.zeros(current_count, dtype=np.float64)

                # In-place generation of time axis:
                # 1. Fill with 0..N-1
                # 2. Scale by step/rate
                # 3. Add start_time

                # Using np.linspace is safer for endpoints but slightly slower?
                # Let's use linspace with 'out'.
                # t_first = start_sample / rate
                # t_last = t_first + (current_count - 1) * step / rate
                # np.linspace(t_first, t_last, current_count, endpoint=True, out=self.t_buffer[:current_count])

                # Actually, precise sample alignment is better with arange logic:
                # t = (start + i*step) / rate

                # View into the result buffer
                target_buffer = self.t_buffer[:current_count]

                # Copy pre-calculated indices 0..N-1
                # We use copyto to avoid allocation
                np.copyto(target_buffer, self.index_buffer[:current_count])

                # Apply scaling and offset in-place
                target_buffer *= (step / self.rate)
                target_buffer += (start_sample / self.rate)

                line.set_data(target_buffer, chunk)

                # Show markers if zooming in enough (step must be 1 to show true samples)
                if step == 1 and chunk.size < 300:
                    line.set_marker('.')
                    line.set_markersize(3)
                else:
                    line.set_marker("")

                global_min_y = min(global_min_y, np.min(chunk))
                global_max_y = max(global_max_y, np.max(chunk))
                has_data = True
            else:
                line.set_data([], [])

        return has_data, global_min_y, global_max_y

    def update_view(self, start_time, width):
        """Core update logic: loads data and sets limits (Constrained Mode)."""
        if self.navigating: return
        self.navigating = True
        try:
            # Clamp width
            width = max(self.min_width_sec, min(width, self.duration_sec))

            # Clamp start time
            start_time = max(0, min(start_time, self.duration_sec))

            has_data, min_y, max_y = self.get_chunk(start_time, width)

            # IMPORTANT: Set limits on the MAIN axes, explicitly using self.ax
            self.ax.set_xlim(start_time, start_time + width)

            # Tight Y-axis scaling logic
            if has_data and max_y > min_y:
                # Symmetric zoom centered at 0
                max_val = max(abs(min_y), abs(max_y))
                min_val = 1670000 # very approximately 1.1mV

                if max_val < min_val:
                    max_val = min_val
                else:
                    max_val *= 1.05

                self.ax.set_ylim(-max_val, max_val)
            else:
                 # Default fallback if no data (-1.0 to 1.0 equivalent)
                 self.ax.set_ylim(-2147483648, 2147483648)

            self.fig.canvas.draw_idle()
        finally:
            self.navigating = False

    def update_slider(self, val):
        """Callback for slider change."""
        if self.navigating: return

        # Get current width from the main property
        current_xlim = self.ax.get_xlim()
        width = current_xlim[1] - current_xlim[0]

        # Fallback for init
        if width <= 0: width = INITIAL_WINDOW_SEC

        self.update_view(val, width)

    def on_scroll(self, event):
        """Handle zoom."""
        if event.inaxes != self.ax: return

        xlim = self.ax.get_xlim()
        cur_width = xlim[1] - xlim[0]

        scale_factor = 0.8 if event.button == 'up' else 1.2
        new_width = cur_width * scale_factor

        # Clamp zoom
        new_width = max(self.min_width_sec, min(new_width, self.duration_sec))

        # Center zoom
        rel_x = (event.xdata - xlim[0]) / cur_width
        new_start = event.xdata - (new_width * rel_x)

        # Clamp bounds
        new_start = max(0, min(new_start, self.duration_sec - new_width))

        # CRITICAL FIX: Set the plot limits FIRST
        self.ax.set_xlim(new_start, new_start + new_width)

        # Then update slider
        self.slider.set_val(new_start)

    def on_xlim_changed(self, ax):
        """Handle external xlim changes (e.g. from Toolbar). Unconstrained load."""
        if self.navigating: return
        self.navigating = True
        try:
            xlim = ax.get_xlim()
            start_time = xlim[0]
            width = xlim[1] - xlim[0]

            # Reload data (No constraints)
            self.get_chunk(start_time, width)

            # Sync slider silently
            if hasattr(self, 'slider'):
                old_eventson = self.slider.eventson
                self.slider.eventson = False
                self.slider.set_val(start_time)
                self.slider.eventson = old_eventson
        finally:
            self.navigating = False

    def on_key(self, event):
        """Handle keyboard shortcuts (Independent Navigation)."""
        if self.navigating: return
        self.navigating = True
        try:
            # Get properties
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            width = xlim[1] - xlim[0]
            height = ylim[1] - ylim[0]

            changed = False

            if event.key == 'right':
                shift = width * 0.25
                self.ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
                changed = True
            elif event.key == 'left':
                shift = width * 0.25
                self.ax.set_xlim(xlim[0] - shift, xlim[1] - shift)
                changed = True
            elif event.key == 'up':
                shift = height * 0.25
                self.ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
                changed = True
            elif event.key == 'down':
                shift = height * 0.25
                self.ax.set_ylim(ylim[0] - shift, ylim[1] - shift)
                changed = True
            elif event.key == 'pagedown': # Zoom In (0.5x width)
                center = (xlim[0] + xlim[1]) / 2
                new_width = width * 0.5
                new_width = max(new_width, self.min_width_sec)
                new_start = center - (new_width / 2)
                # Clamp start
                new_start = max(0, min(new_start, self.duration_sec - new_width))
                self.ax.set_xlim(new_start, new_start + new_width)
                changed = True
            elif event.key == 'pageup': # Zoom Out (2x width)
                center = (xlim[0] + xlim[1]) / 2
                new_width = width * 2.0
                new_width = min(new_width, self.duration_sec)
                new_start = center - (new_width / 2)
                # Clamp start
                new_start = max(0, min(new_start, self.duration_sec - new_width))
                self.ax.set_xlim(new_start, new_start + new_width)
                changed = True

            if changed:
                # Reload data for new X view
                # We need new xlim
                new_xlim = self.ax.get_xlim()
                self.get_chunk(new_xlim[0], new_xlim[1] - new_xlim[0])

                # Sync slider silently
                if hasattr(self, 'slider'):
                    old_eventson = self.slider.eventson
                    self.slider.eventson = False
                    self.slider.set_val(new_xlim[0])
                    self.slider.eventson = old_eventson

                self.fig.canvas.draw_idle()

        finally:
             self.navigating = False

        # Space Bar (Reset Zoom) - Only reset Y axis to fit visible data (Auto-scale)
        if event.key == ' ':
            xlim = self.ax.get_xlim()
            width = xlim[1] - xlim[0]
            self.update_view(xlim[0], width)

    def on_select(self, eclick, erelease):
        """Handle rectangle selection."""
        if self.navigating: return

        # Calculate new ranges from the selection
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        start_time = min(x1, x2)
        end_time = max(x1, x2)
        width = end_time - start_time

        # Enforce minimum width
        if width < self.min_width_sec:
            center = (start_time + end_time) / 2
            width = self.min_width_sec
            start_time = center - width / 2

        # Clamp start time
        start_time = max(0, min(start_time, self.duration_sec - width))

        # Update view (which reloads data)
        # Note: update_view handles height scaling automatically based on data,
        # but if we want to respect the user's Y selection we might need to override.
        # However, for audio, usually we want to see the full amplitude or auto-scaled.
        # The user asked for "arbitrary zooming be implemented", which implies Y might be important.
        # But our `update_view` forces Y symmetry currently.
        # Let's trust `update_view` for now as it handles the complexity of data loading.
        # If we really want to zoom to the Y rectangle, we would set ylim manually.
        # Given the previous context was about "Zoom to Rectangle" interfering with "movement and zoom control",
        # let's try to match the "Zoom to Rectangle" behavior which DOES set specific X/Y limits.

        # However, our update_view logic re-calculates Y based on data in the new X range...
        # Let's modify the flow slightly: use update_view for X, then apply Y if it makes sense?
        # Actually, standard audio visualization often keeps Y symmetric or 0-1.
        # If the user draws a small box on a peak, they expect to zoom into that peak (Y-wise too).

        # So let's set limits directly, similar to on_scroll/on_key, but for both axes.

        if self.navigating: return
        self.navigating = True
        try:
             # X Axis Logic
             new_width = max(self.min_width_sec, min(width, self.duration_sec))
             # If user selected < min width, we centered it above.

             self.get_chunk(start_time, new_width)

             self.ax.set_xlim(start_time, start_time + new_width)

             # Y Axis Logic
             y_min = min(y1, y2)
             y_max = max(y1, y2)

             self.ax.set_ylim(y_min, y_max)

             self.fig.canvas.draw_idle()

             # Sync Slider
             if hasattr(self, 'slider'):
                 old_eventson = self.slider.eventson
                 self.slider.eventson = False
                 self.slider.set_val(start_time)
                 self.slider.eventson = old_eventson

        finally:
            self.navigating = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linux Audio Waveform Visualizer 2026 (mmap)")
    parser.add_argument('files', nargs='+', help="Input .bin files (int32)")
    parser.add_argument('--rate', type=int, default=48000, help="Sample rate (Hz)")
    parser.add_argument('--min-zoom-samples', type=int, default=100, help="Minimum samples to show when zoomed in")
    args = parser.parse_args()

    app = WaveformVisualizer(args.files, args.rate, args.min_zoom_samples)
