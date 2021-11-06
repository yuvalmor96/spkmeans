#ifndef SPKMEANS_H_
#define SPKMEANS_H_

void readfile();
void init_cendroids(int clusters_num);
double distance(double *v1, double *v2);
int min_dist_centroid(double *v);
void vector_to_cluster(int clusters_num);
double* cluster_to_centroid(int index);
int areequal(double* arr1, double* arr2);
int update_centroids();
double**  calccentroids(int max_iter);
double** matMult(double** A, double** B);
double** weightedAdjMat();
double weightedAdjMat_calc(double* v1,double* v2);
double** Lnorm();
void printMat(double** Mat, int rowNum, int colNum);
double** diagDegMat();
void calcRotationMat(double** P,double c, double s, int row, int column);
double calc_theta(double **Mat, int i, int j);
double calc_t(double theta);
double calc_c(double t);
double calc_s(double t, double c);
int isConverged(double** A, double** Aprime);
void calcAprime(double** A, double** Aprime, int i, int j, double c, double s);
double** calcJacobi(double **A);
int eigenComperator(const void *a, const void *b);
void sortEigenVectors();
int eigengapHeuristic();
void createTMat();
void fullSpectral();
double** transpMat(double **Mat);
void assertmsg(int x);
void freearray(double** array, int length);


#endif