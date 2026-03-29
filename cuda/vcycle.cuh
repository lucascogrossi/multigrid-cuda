#ifndef VCYCLE_CUDA_H
#define VCYCLE_CUDA_H

#include <functional>

#include "grid_device.cuh"
#include "multigrid_utils.cuh"

// Recebe a hierarquia de grids pre alocadas
__host__ void v_cycle(std::vector<Grid2D*>& grids) {

    // Descida ate o penultimo nivel da grid
    for (int i = 0; i < grids.size() - 1; i++) {

        dim3 numThreadsPerBlock(16, 16);
        dim3 numBlocks((grids[i]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                       (grids[i]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

        // 1. Pre suavizacao
        for (int k = 0; k < 2; k++) {
            gauss_seidel_rb_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], 0);
            gauss_seidel_rb_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], 1);
        }

        // 2. Calcula residuo
        compute_residual_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], grids[i]->r);

        // 3. Restrict
        restriction_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i]->r, grids[i+1]->f, grids[i]->nx, grids[i]->ny);

        // 4. zera u do prox nivel
        cudaMemset(grids[i+1]->u, 0, (grids[i+1]->nx+1)*(grids[i+1]->ny+1)*sizeof(double));
    }

    // 5. Calcula direto na malha mais grossa q existe
    solve_coarse(grids.back(), 1);

    // Subida ate o nivel mais fino (primeiro nivel)
    for (int i = grids.size()-2; i >= 0; i--) {

        dim3 numThreadsPerBlock(16, 16);
        dim3 numBlocks((grids[i]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                       (grids[i]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
        
        // 6. Prolongation
        prolongation_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i+1]->u, grids[i]->e, grids[i+1]->nx, grids[i+1]->ny);

        // 7. Correct
        correct_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], grids[i]->e);

        // 8. Pos suavizacao
        for (int k = 0; k < 2; k++) {
            gauss_seidel_rb_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], 0);
            gauss_seidel_rb_kernel<<<numThreadsPerBlock, numBlocks>>>(grids[i], 1);
        }
    }   
}

#endif