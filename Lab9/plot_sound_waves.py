from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters from the C++ simulation
L = 10.0  # Room side length in meters
N = 100   # Grid points per dimension

# Load simulation data
def load_frame(frame_number):
    """Load a single frame from the simulation data."""
    filepath = f"simulation_data/frame_{frame_number}.dat"
    try:
        data = np.loadtxt(filepath)
        data = np.flipud(data)
    except OSError:
        return None
    return data

# Get all available frames
def get_available_frames():
    """Get list of all available frame numbers."""
    data_dir = "simulation_data"
    frame_files = [f for f in os.listdir(data_dir) if f.startswith("frame_") and f.endswith(".dat")]
    
    # Extract frame numbers and sort them
    frame_numbers = []
    for f in frame_files:
        frame_num_str = f.split('_')[1].split('.')[0]
        frame_numbers.append(int(frame_num_str))
    
    return sorted(frame_numbers)

# Get frames
frame_numbers = get_available_frames()
print(f"Found {len(frame_numbers)} frames")
print(f"Frame range: {frame_numbers[0]} to {frame_numbers[-1]}")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Load first frame to get data range for consistent colormap
first_frame = load_frame(frame_numbers[0])

# Create the initial image
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

im = ax.imshow(first_frame, extent=[0, L, 0, L], origin='lower', cmap='seismic', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Wave Amplitude')

# Set labels and title
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
title = ax.set_title(f'2D Sound Wave Propagation - Frame {frame_numbers[0]}')

# Animation update function
def update(frame_idx):
    """Update function for animation."""
    frame_num = frame_numbers[frame_idx]
    data = load_frame(frame_num)
    
    if data is not None:
        im.set_array(data)
        title.set_text(f'2D Sound Wave Propagation - Frame {frame_num}')
    
    return [im, title]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(frame_numbers), interval=50, blit=True, repeat=True)

# Show the plot
plt.show()

# Save the animation
while True:
    save_animation = input("Do you want to save the animation? y/n: ")
    if save_animation.lower() in ['y', 'n']:
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

if save_animation.lower() == 'y':
    print("Saving animation...")
    filename = f"simulation_data/sound_wave_animation_{str(datetime.now()).replace(':', '').replace(' ', '_')}.mp4"
    ani.save(filename, writer='ffmpeg', fps=20, dpi=100)
    print(f"Animation saved as '{filename}'")
else:
    print("Animation not saved.")

# Optionally remove individual frame files after creating the animation
while True:
    delete_files = input("Do you want to delete the individual frame files? y/n: ")
    if delete_files.lower() in ['y', 'n']:
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

if delete_files.lower() == 'y':
    for i in range(frame_numbers[0], frame_numbers[-1] + 1):
        os.remove(f"simulation_data/frame_{i}.dat")
else:
    print("Frame files retained.")