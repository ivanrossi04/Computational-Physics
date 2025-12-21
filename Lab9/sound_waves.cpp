// !V1
/*
 * Compute the propagation of sound waves in a 2D space by solving the wave equation numerically with finite differences.
 * The simulation is performed in a square room of side L with rigid walls with Neumann boundary conditions.
 * 
 * The room layout is defined in an external text file "room_layout.txt" where walls and free space are represented by different characters:
 *  O -> wall
 *  . -> free space
 * 
 * The initial condition is a Gaussian wave packet centered at (x0, y0) with amplitude A and frequency f.
 * The simulation parameters (A, f, x0, y0, N, N_steps, DeltaT) can be set via command line arguments.
 * 
 * Frames are saved in the simulation folder at regular intervals for visualization with an external tool.
*/

#include <iostream>
#include <fstream>
#include <vector>

// Auxiliary libraries
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <chrono>

/**
 * @brief Save the current frame to a file, the folder "simulation_data" must exist 
 * 
 * @param u The 2D array representing the wave state
 * @param size The size of the 2D array (number of grid points per dimension), if -1 saves only the initial condition
 * @param filepath The filepath where to save the frame
 */
template<typename T>
void save_frame(const std::vector<std::vector<T>>& u, int size, const char* filepath);

/**
 * @brief Update the wave state using finite difference methods
 * 
 * @param u_prev The wave state at the previous time step
 * @param u_curr The wave state at the current time step
 * @param u_next The wave state at the next time step (the computation data will be stored here)
 * @param grad_mask The gradient mask representing the room layout and wall types
 * @param size The size of the 2D arrays (number of grid points per dimension)
 * @param Courant_sq The square of the Courant number (c * DeltaT / DeltaX)^2
 */
void update_wave(std::vector<std::vector<double>>* u_prev, std::vector<std::vector<double>>* u_curr, std::vector<std::vector<double>>* u_next, std::vector<std::vector<int>>* grad_mask, int size, double Courant_sq);

int main(int argc, char* argv[]) {
    // Simulation parameters
    int N = 100; // Number of grid points per dimension
    int N_steps = 2001; // Number of time steps

    const double L = 10.0; // Room side length in meters

    double DeltaT = 1e-4; // Time step (s)

    // Physical parameters
    const double c = 300.0; // Speed of sound (m/s)

    // Wave parameters
    double A = 1.0; // Amplitude of the wave
    double frequency = 5.0; // Frequency of the wave (Hz)
    double x0 = L / 2; // Initial x position of the wave center (m)
    double y0 = L / 2; // Initial y position of the wave center (m)

    if(argc > 1) {
        A = std::stod(argv[1]);
        if(argc > 2) frequency = std::stod(argv[2]);
        if(argc > 3) x0 = std::stod(argv[3]);
        if(argc > 4) y0 = std::stod(argv[4]);
        if(argc > 5) N = std::stoi(argv[5]);
        if(argc > 6) N_steps = std::stoi(argv[6]);
        if(argc > 7) DeltaT = std::stod(argv[7]);
    }

    int N_save = 3; // (1 / (DeltaT * 1000)); // Save a frame every N_save time steps (assuming 60 FPS for visualization slowed down)
    N_save -= N_save % 3; // Ensure N_save is a multiple of 3 to align with the update cycle

    std::cout << "Saving a frame every " << N_save << " time steps.\n";

    const double DeltaX = L / (N - 1); // Spatial step (m)
    
    const double Courant_sq = pow(c * DeltaT / DeltaX, 2); // Courant number squared
    std::cout << "Courant squared: " << Courant_sq << "\n";

    if(Courant_sq >= 0.5) {
        std::cerr << "Warning: Courant condition not satisfied, the simulation may be unstable! Aborting simulation.\n";
        return -1;
    }

    /*
    * From a room layout file, create an N x N matrix representing the room layout:
    * In the layout file, use the following symbols:
    *   O -> wall
    *   . -> free space
    * The create matrix function should read the layout file and scale it to fit the N x N grid.
    */
    std::ifstream layout_file("room_layout.txt");
    if(!layout_file.is_open()) {
        std::cerr << "Error opening room layout file. Make sure 'room_layout.txt' exists.\n";
        return -1;
    }
    int layout_size;;
    layout_file >> layout_size;

    // The ratio corresponds to the "resolution" of the layout file compared to the simulation grid
    int layout_ratio = N / layout_size;

    std::vector<std::vector<bool>> room_layout = std::vector<std::vector<bool>>(N, std::vector<bool>(N, 0));
    for (int i = 0; i < layout_size; ++i) {
        for(int j = 0; j < layout_size; ++j) {
            char ch;
            layout_file >> ch;

            // Fill the corresponding block in the room_layout matrix
            for(int n = 0; n < layout_ratio; ++n) {
                for(int m = 0; m < layout_ratio; ++m) {
                    int x = i * layout_ratio + n;
                    int y = j * layout_ratio + m;
                    room_layout[x][y] = (ch == 'O'); // 1 for wall, 0 for free space
                }
            }
        }
    }
    // Save the room layout for verification
    save_frame(room_layout, N, "simulation_data/room_layout.dat");

    /*
    * Create a grid of size N x N to represent the room layout and the walls with Neumann boundary conditions (rigid walls).
    * The different values in the matrix represent different types of barrier:
    *   0 -> free space
    *   1 -> inner wall
    *   2 -> upper wall
    *   3 -> right wall
    *   4 -> lower wall
    *   5 -> left wall
    *   6 -> upper-right corner
    *   7 -> lower-left corner
    *   8 -> upper-left corner
    *   9 -> lower-right corner
    */
    std::vector<std::vector<int>> grad_mask(N, std::vector<int>(N));
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(room_layout[i][j]) grad_mask[i][j] = 1; // inner wall (solid cell)
            else {
                // Check adjacent cells to determine boundary type
                bool wall_up = (i == 0) || (i > 0 && room_layout[i-1][j]);
                bool wall_down = (i == N-1) || (i < N-1 && room_layout[i+1][j]);
                bool wall_left = (j == 0) || (j > 0 && room_layout[i][j-1]);
                bool wall_right = (j == N-1) || (j < N-1 && room_layout[i][j+1]);
                
                // Count walls
                int wall_count = (wall_up ? 1 : 0) + (wall_down ? 1 : 0) + 
                                (wall_left ? 1 : 0) + (wall_right ? 1 : 0);
                
                if(wall_count >= 2) {
                    // Corner detection (two or more walls meet)
                    if(wall_up && wall_right) grad_mask[i][j] = 6; // upper-right corner
                    else if(wall_down && wall_right) grad_mask[i][j] = 7; // lower-right corner
                    else if(wall_down && wall_left) grad_mask[i][j] = 8; // lower-left corner
                    else if(wall_up && wall_left) grad_mask[i][j] = 9; // upper-left corner
                    else grad_mask[i][j] = 0; // opposite walls (free space, no special treatment)
                } else if(wall_count == 1) {
                    // Edge detection (single wall)
                    if(wall_up) grad_mask[i][j] = 2; // upper wall
                    else if(wall_down) grad_mask[i][j] = 3; // lower wall (was labeled "right" before)
                    else if(wall_left) grad_mask[i][j] = 4; // left wall (was labeled "lower" before)
                    else if(wall_right) grad_mask[i][j] = 5; // right wall (was labeled "left" before)
                } else {
                    grad_mask[i][j] = 0; // free space
                }
            }
        }
    }
    // Save the gradient mask for verification
    save_frame(grad_mask, N, "simulation_data/grad_mask.dat");

    // Start timer here
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Configure the first state of the wave
    std::vector<std::vector<double>> u1(N, std::vector<double>(N)); // Wave state at time step 0
    for(int i = 0; i < N; ++i) {
        double x = i * DeltaX;
        for(int j = 0; j < N; ++j) {
            double y = j * DeltaX;
            // Simulation parameters: Gaussian initial condition centered in the room
            if(!room_layout[i][j]) u1[i][j] = A * exp(-(pow(x - x0, 2) + pow(y - y0, 2))/(0.1));
        }
    }
    save_frame(u1, N, "simulation_data/initial_frame.dat"); // Save the initial frame

    // Initialize u2 using a Taylor expansion (assuming initial velocity is zero)
    std::vector<std::vector<double>> u2(N, std::vector<double>(N)); // Wave state at time step 1
    for(int i = 1; i < N - 1; ++i) {
        for(int j = 1; j < N - 1; ++j) {
            if(grad_mask[i][j] == 0) {
                double laplacian = (u1[i+1][j] + u1[i-1][j] + u1[i][j+1] + u1[i][j-1] - 4 * u1[i][j]);
                u2[i][j] = u1[i][j] + 0.5 * Courant_sq * laplacian;
            }
        }
    }

    // Apply Neumann boundary conditions to u2
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            switch(grad_mask[i][j]) {
                case 1: // inner wall
                    u2[i][j] = u2[i][j];
                    break;
                case 2: // upper wall
                    u2[i][j] = u2[i+1][j];
                    break;
                case 3: // lower wall
                    u2[i][j] = u2[i-1][j];
                    break;
                case 4: // left wall
                    u2[i][j] = u2[i][j+1];
                    break;
                case 5: // right wall
                    u2[i][j] = u2[i][j-1];
                    break;
                case 6: // upper-right corner
                    u2[i][j] = u2[i+1][j-1];
                    break;
                case 7: // lower-right corner
                    u2[i][j] = u2[i-1][j-1];
                    break;
                case 8: // lower-left corner
                    u2[i][j] = u2[i-1][j+1];
                    break;
                case 9: // upper-left corner
                    u2[i][j] = u2[i+1][j+1];
                    break;
                default:
                    break; // free space, already handled
            }
        }
    }

    // Time evolution loop
    std::vector<std::vector<double>> tmp(N, std::vector<double>(N));
    for(int n = 0; n < N_steps; n += 3) {
        /*
        * Update the wave states in a cyclic manner to avoid shifting data in memory by computing and then swapping pointers.
        * After three updates, u2 will contain the last computed state and u1 the previous one.
        */
        update_wave(&u1, &u2, &tmp, &grad_mask, N, Courant_sq);
        update_wave(&u2, &tmp, &u1, &grad_mask, N, Courant_sq);
        update_wave(&tmp, &u1, &u2, &grad_mask, N, Courant_sq);

        // Save frame at regular intervals
        if (n % N_save == 0) save_frame(u2, N, ("simulation_data/frame_" + std::to_string(n / N_save) + ".dat").c_str());
    }

    // Stop timer here and print the elapsed time
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Simulation completed in " << elapsed_seconds.count() << "s\n";

    return 0;
}

void update_wave(std::vector<std::vector<double>>* u_prev, std::vector<std::vector<double>>* u_curr, std::vector<std::vector<double>>* u_next, std::vector<std::vector<int>>* grad_mask, int size, double Courant_sq) {
    // Update the interior points using finite difference (the cycle starts from 1 to N-1 to avoid boundary points)
    for(int i = 1; i < size - 1; ++i) {
        for(int j = 1; j < size - 1; ++j) {
            if((*grad_mask)[i][j] == 0) { // Free space
                double laplacian = ((*u_curr)[i+1][j] + (*u_curr)[i-1][j] + (*u_curr)[i][j+1] + (*u_curr)[i][j-1] - 4 * (*u_curr)[i][j]);
                (*u_next)[i][j] = 2 * (*u_curr)[i][j] - (*u_prev)[i][j] + Courant_sq * laplacian;
            }
        }
    }

    // Apply Neumann boundary conditions
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            switch((*grad_mask)[i][j]) {
                case 1: // inner wall
                    (*u_next)[i][j] = (*u_curr)[i][j];
                    break;
                case 2: // upper wall
                    (*u_next)[i][j] = (*u_next)[i+1][j];
                    break;
                case 3: // lower wall
                    (*u_next)[i][j] = (*u_next)[i-1][j];
                    break;
                case 4: // left wall
                    (*u_next)[i][j] = (*u_next)[i][j+1];
                    break;
                case 5: // right wall
                    (*u_next)[i][j] = (*u_next)[i][j-1];
                    break;
                case 6: // upper-right corner
                    (*u_next)[i][j] = (*u_next)[i+1][j-1];
                    break;
                case 7: // lower-right corner
                    (*u_next)[i][j] = (*u_next)[i-1][j-1];
                    break;
                case 8: // lower-left corner
                    (*u_next)[i][j] = (*u_next)[i-1][j+1];
                    break;
                case 9: // upper-left corner
                    (*u_next)[i][j] = (*u_next)[i+1][j+1];
                    break;
                default:
                    break; // free space, already handled
            }
        }
    }
}

template<typename T>
void save_frame(const std::vector<std::vector<T>>& u, int size, const char* filepath) {
    static std::ofstream file;
    file.open(filepath);
    
    if(!file.is_open()) {
        std::cerr << "Error opening file for writing frame at: " << filepath << "\n";
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << u[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}
