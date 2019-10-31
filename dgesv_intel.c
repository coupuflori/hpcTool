#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include "mkl_lapacke.h"

double *generate_matrix(int size, int seed)
{
  int i;
  double *matrix = (double *)malloc(sizeof(double) * size * size);
  srand(seed);

  for (i = 0; i < size * size; i++)
  {
    matrix[i] = rand() % 100;
  }

  return matrix;
  free(matrix);
}

void print_matrix(const char *name, double *matrix, int size)
{
  int i, j;
  printf("matrix: %s \n", name);

  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      printf("%f ", matrix[i * size + j]);
    }
    printf("\n");
  }
}

int check_result(double *bref, double *b, int size) {
  int i;
  for(i=0;i<size*size;i++) {
    if(abs(bref[i] - b[i]) > 0.000005) return 0; 
  } 
  return 1;
}

//converts a vector to a matrix
void vectToMat(double *vect, double **matRes, int size) {
  int i;
  int j;

  for(i=0; i<size; i++) {
    for(j=0; j<size; j++) {
        matRes[i][j] = vect[i * size + j];
    }
  }

}

//Used to multiply a given line by a given value (val)
void dilatation(double ** mat, int line, int size, double val) {

    for(int j=0; j<size; j++) {
        mat[line][j] = val * mat[line][j];
    }
}

//Given two lines, computes line1 <- line1 + k*line2 
void transvection(double ** mat, int size, double k, int line1, int line2) {
  double startTrans = omp_get_wtime();
  #pragma omp parallel for
  #pragma code_align 32
  #pragma ivdep
    for(int j=0; j<size; j++) {
        mat[line1][j] = mat[line1][j] + (k * mat[line2][j]);
    }
  double endTrans = omp_get_wtime();
  //printf("transvection : %f \n", startTrans-endTrans);
}

//Used to put the pivot values to 1 and to put 0 values underneath it
void addZeroUnderPivot(double ** mat, double ** matId, int size, int iPivot) {
  int i;
  double pivot; //val for dilatation
  double val; //val for transvection

  //Pivot is the diagonal value
  pivot = mat[iPivot][iPivot];

  if(pivot == 0) {
    exit(-1);
  }

  //if index is the one before the last line of the matrix, we only need to put the pivot valut at 1
  if(iPivot == (size - 1)) {
    //Puts the pivot value at 1 with dilatation of the line of the original matrix
    dilatation(mat, iPivot, size, (1 / pivot));
    //Puts the pivot value at 1 with dilatation of the line of the identity matrix
    dilatation(matId, iPivot, size, (1 / pivot));
  } else { //else we also need to put 0 underneath the pivot

    dilatation(mat, iPivot, size, (1 / pivot));
    dilatation(matId, iPivot, size, (1 / pivot));

    //Used to put 0 underneath the pivot thanks to the transvection operation on the original matrix and on the identity matrix
    #pragma omp parallel for private(val)
    for(i=(iPivot + 1); i<size; i++) {
      val = mat[i][iPivot];
      transvection(mat, size, (- val), i, iPivot);
      transvection(matId, size, (- val), i, iPivot);
    }
  
  }
}

//Used to put the pivot values to 1 and to put 0 values over it
void addZeroOverPivot(double ** mat, double ** matId, int n, int iPivot) {
  int i;
  double k;   
  
  #pragma omp parallel for private(k) 
  for(i=iPivot;i>0;i--) {
    
    k = mat[i-1][iPivot];

    //Transvection used on the original matrix and on the identity matrix
    transvection(mat, n, (- k), i-1, iPivot);
    transvection(matId, n, (- k), i-1, iPivot);
    
  }
}

//Used to create an identity matrix 
void matrixId(double **matRes, int size) {
  int i; 
  int j;  

  for (i=0;i<size;i++) {
    for (j=0;j<size;j++) {
      if (i==j) {
        matRes[i][j] =  1;
      } else {
        matRes[i][j] = 0;
      }
    }
  }
}

//used to reverse a matrix using Gauss method
void reverseGauss(double ** mat, double ** matId, int size) {
  int i;

  for (i=0; i<size; i++) {
    //Lower triangularization
    addZeroUnderPivot(mat,matId,size,i); 
  }
  for(i=(size)-1;i>0;i--) {
    //Upper triangularization
    addZeroOverPivot(mat,matId,size,i);    
  }
}

void transpose(double **A, double **B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j][i] = A[i][j];
        }
    }
}

//multiplication of two matrices to obtain result X
void computeX(double **reverseA, double **matB, double **matRes, int size) {
  int i;
  int j;
  int k;
  double **transB;

  transB = malloc(size * sizeof(double *));
  for(i=0; i<size; i++) {
    transB[i] = malloc(size * sizeof(double));
  }

  transpose(matB, transB, size);

  #pragma omp parallel for private(j,k)
  for(i=0; i<size; i++) {
    for(j=0; j<size; j++) {
      for(k=0; k<size; k++) {
        matRes[i][j] += reverseA[i][k] * transB[j][k];
      }            
    }
  }
}

//converts matrice into vector
void matToVect(double **mat, double *vectRes, int size) {
  for(int i=0; i<size; i++) {
    for(int j=0; j<size; j++) {
      vectRes[i * size + j] = mat[i][j];
    }
  }
}

void my_dgesv(double *vectA, double *vectB, double *vectX, int size) {
  double **matA = malloc(size * sizeof(double *));
  double **matB = malloc(size * sizeof(double *));
  double **matId = malloc(size*(sizeof(double *)));
  double **matX = malloc(size*(sizeof(double *)));
    for (int i=0; i<size;i++) {
        matA[i] = malloc(size * sizeof(double));
        matB[i] = malloc(size * sizeof(double));
        matId[i] = malloc(size*(sizeof(double)));
        matX[i] = malloc(size*(sizeof(double)));
  }

  vectToMat(vectA, matA, size);
  matrixId(matId,size);
  reverseGauss(matA, matId, size);
  vectToMat(vectB, matB, size);
  computeX(matId, matB, matX, size);
  matToVect(matX, vectX, size);

 for (int i=0; i<size; ++i) {
  free(matA[i]);
  free(matB[i]);
  free(matId[i]);
  free(matX[i]);
 } 
 free(matA); 
 free(matB); 
 free(matId); 
 free(matX); 

}


int main(int argc, char *argv[]) {
  int size = atoi(argv[1]);
  double *a, *aref;
  double *b, *bref;

  a = generate_matrix(size, 1);
  aref = generate_matrix(size, 1);        
  b = generate_matrix(size, 2);
  bref = generate_matrix(size, 2);
        
  //print_matrix("A", a, size);
  //print_matrix("B", b, size);

  // Using MKL to solve the system
  //MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
  //MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

  //clock_t tStart = clock();
 // info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
  //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  double *vectX = malloc(sizeof(double) * size * size);

  clock_t tStart = clock();          
  my_dgesv(a, b, vectX, size);
  //printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        
  /*if (check_result(bref,vectX,size)==1)
    printf("Result is ok!\n");
      else    
        printf("Result is wrong!\n");*/
        
  //print_matrix("X", vectX, size);
  //print_matrix("Xref", bref, size);

  free(vectX);
  free(a);
  free(aref);
  free(b);
  free(bref);

  return 0;
}
