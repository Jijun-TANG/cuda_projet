#include<stdio.h>
#include <chrono>
#define SIZE_CR 512
#define SIZE 1024
#define SIZE_ODD 1023
#define ODD2 512
//__global__ void pcr_odd(float *a,float *b,float *c,float *d,float *rhs,int N);
__global__ void pcr(float *a,float *b,float *c,float *d,float *rhs,int N);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ void thomas(float *a,float *b,float *c,float *d,float *rhs,int N){
    //forward elimination
    c[0]=c[0]/b[0];
    d[0]=d[0]/b[0];
    b[0]=1.0;
    for(int i=1;i<N;++i){
        float t1=b[i]-c[i-1]*a[i];
        c[i]=c[i]/(t1);
        d[i]=(d[i]-d[i-1]*a[i])/(t1);
    }
    //backward substitution
    rhs[N-1]=d[N-1];
    for(int i=N-2;i>=0;--i){
        rhs[i]=d[i]-c[i]*rhs[i+1];
    }
}

__global__ void cr(float *a,float *b,float *c,float *d,float *rhs,int N){
    __shared__ float a_i[SIZE_CR];    __shared__ float b_i[SIZE_CR];
    __shared__ float c_i[SIZE_CR];    __shared__ float d_i[SIZE_CR];

    int coeff=2; //stride number
    int i=(threadIdx.x+1)*coeff-1; //the first step of reduction there will be N/2 threads and each thread will reduce the even indexes of equations e2,e4,e6,...eN (rhs[1],rhs[3],rhs[5],...rhs[N-1])
    float k1=a[i]/b[i-1],k2;

    // first reduction (first step)
    if(i<N-1){
        k2=c[i]/b[i+1];}
    else{k2=0.0;}

    a_i[threadIdx.x]=-a[i-1]*k1;
    if(i!=N-1){ //general cases
        b_i[threadIdx.x]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    else{ //special case (last equation)
        b_i[threadIdx.x]=b[i]-c[i-1]*k1;
        c_i[threadIdx.x]=0.0;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1;
    }
    __syncthreads();
    int num=SIZE_CR/2;        //number of threads needed

    //forward reduction (second step to log2(N)-1 th step)
    do{
        i=(threadIdx.x+1)*coeff-1;
        if(threadIdx.x<num){
            k1=a_i[i]/b_i[i-coeff/2];
            a_i[i]=-a_i[i-coeff/2]*k1;

            if(threadIdx.x!=num-1){ //general case
                k2=c_i[i]/b_i[i+coeff/2];
                b_i[i]=b_i[i]-c_i[i-coeff/2]*k1-a_i[i+coeff/2]*k2;
                c_i[i]=-c_i[i+coeff/2]*k2;
                d_i[i]=d_i[i]-d_i[i-coeff/2]*k1-d_i[i+coeff/2]*k2;
            }
            else{ //specific case (last equation)
                b_i[i]=b_i[i]-c_i[i-coeff/2]*k1;
                c_i[i]=0.0;
                d_i[i]=d_i[i]-d_i[i-coeff/2]*k1;
            }
            //printf("\nIN LOOPIdx: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
        }
        num=num/2; //diminution of the number of threads needed
        coeff=coeff*2; //augmentation of the shifting
    }while(num>1);

    //log2(N) th step
    //solve the 2 unknowns Xn/2(rhs[(N/2-1]) and Xn(rhs[N-1]) and the other 2 in the middle of them ( (0,XN/2) & (XN/2,XN) )
    //by substitution
    if(threadIdx.x==0){
        if(b_i[i]==0){                // if b[N/2-1] = 0 
            rhs[4*i+3]=d_i[i]/c_i[i];
            rhs[2*i+1]=(d_i[2*i+1]-b_i[2*i+1]*rhs[4*i+3])/a_i[2*i+1];
        }
        else if(a_i[2*i+1]==0){       // if a[N-1] = 0 the coefficient of the XN/2 
            rhs[4*i+3]=d_i[2*i+1]/b_i[2*i+1];
            rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[4*i+3])/b_i[2*i+1];
        }
        else{                         //normal routine for solving the systems of 2 unknowns
            rhs[4*i+3]=(d_i[i]-d_i[2*i+1]*b_i[i]/a_i[2*i+1])/(c_i[i]-b_i[2*i+1]*b_i[i]/a_i[2*i+1]);
            rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[4*i+3])/b_i[i];
        }
        //printf("first result check %d %d\n%f\n%f\n",2*i+1,4*i+3,rhs[2*i+1],rhs[4*i+3]);

        // solve the precedente matrix (normal routine for solving the systems of 4 unknowns given that we know 2 of them)
        if(N>4){
            rhs[3*i+2]=(d_i[(3*i+1)/2]-a_i[(3*i+1)/2]*rhs[2*i+1]-c_i[(3*i+1)/2]*rhs[4*i+3])/b_i[(3*i+1)/2];
            rhs[i]=(d_i[i/2]-c_i[i/2]*rhs[2*i+1])/b_i[i/2];
        }
        //printf("second result check %d %d\n%f\n%f\n",i,3*i+2,rhs[i],rhs[3*i+2]);
    }
    ///check the first 4 solutions of system
    /*
    for(int j=0;j<N;++j){
        printf("111X%d = %f\n",j+1,*(rhs+j));
    }*/


    //backward substitution (log2(N)+1 to 2log2(N)-1 th step)
    num=num*4; //augmentation of the number of threads needed we multiply by 4 because we did one more time 'num/=2' and 'coeff*=2' in do while loop
    coeff/=4; //diminution of the stride

    /*
    if(threadIdx.x==0){
        printf("maximum num threads value %d coeff value %d\n",num,coeff);
    }*/

    while(num<SIZE_CR){
        i=-1+(coeff/2)+threadIdx.x*coeff; //i maps to indice correspondant of backward substitution
        if(threadIdx.x<num){
            if(threadIdx.x==0){ //specific case (first equation, there is no a_i)
                rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[2*i+1+coeff])/b_i[i];
            }
            else{ //general case
                rhs[2*i+1]=(d_i[i]-a_i[i]*rhs[2*i+1-coeff]-c_i[i]*rhs[2*i+1+coeff])/b_i[i];
            }
            //for testing
            //printf("backward in loop value rhs[%d] = %f\n",2*i+1,rhs[2*i+1]);
        }
        __syncthreads();
        num*=2;
        coeff/=2;
    }
    //check backward propogation
    //printf("check backward propogation rhs[%d] = %f\nrhs[%d] = %f\n",threadIdx.x,*(rhs+threadIdx.x),threadIdx.x+1,*(rhs+threadIdx.x+1));
    if(threadIdx.x<SIZE_CR){
        i=2*threadIdx.x;
        if(i==0){ //specific case (first equation)
            rhs[i]=(d[i]-c[i]*rhs[i+1])/b[i];
        }
        else{ //general case
            rhs[i]=(d[i]-c[i]*rhs[i+1]-a[i]*rhs[i-1])/b[i];
        }
    }
}
/*
void test_8_cr(){
    int D=8;
    //rhs=(float*)malloc(d*sizeof(float));
    float a[8]={0.0,1.0,7.0,2.0,2.0,3.0,-1.0,-1.0};
    float b[8]={1.0,1.0,1.0,11.0,3.0,1.0,2.0,2.0};
    float c[8]={2.0,10.0,2.0,1.0,7.0,2.0,2.0,0.0};
    float d[8]={4.0,14.0,26.0,25.0,0.0,2.0,1.0,3.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    cr<<<1,4>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));

    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}


void test_16_cr(){
    int D=16;
    //rhs=(float*)malloc(d*sizeof(float));
    float a[16]={0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
    float b[16]={2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};
    float c[16]={-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0};
    float d[16]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    cr<<<1,8>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));

    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}*/

void test_64_cr(){
    int D=64;
    //rhs=(float*)malloc(d*sizeof(float));
    float a[64]={0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
    float b[64]={2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};
    float c[64]={-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0};
    float d[64]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    pcr<<<2,32>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));

    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}



__global__ void pcr(float *a,float *b,float *c,float *d,float *rhs,int N){
    __shared__ float a_i[SIZE_CR];    __shared__ float b_i[SIZE_CR];
    __shared__ float c_i[SIZE_CR];    __shared__ float d_i[SIZE_CR];
    int i=2*threadIdx.x+blockIdx.x;
    float k1=i==0?0.0:a[i]/b[i-1],k2;

    // First reduction (first step)
    if(i<N-1){
        k2=c[i]/b[i+1];}
    else{k2=0.0;}

    if(i==0){ // specific case (first equation)
        b_i[threadIdx.x]=b[i]-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i+1]*k2;
        a_i[threadIdx.x]=0.0;
    }
    else if(i==N-1){ // specific case (last equation)
        a_i[threadIdx.x]=-a[i-1]*k1;
        b_i[threadIdx.x]=b[i]-c[i-1]*k1;
        c_i[threadIdx.x]=0.0;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1;
    }
    else{ // general case
        a_i[threadIdx.x]=-a[i-1]*k1;
        b_i[threadIdx.x]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    //printf("Phase1 check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
    __syncthreads();
    int coeff=1; //stride for correspondance between thread and equation
    float ta;
    float tb;
    float tc;
    float td;
    //printf("before loop %d\n",threadIdx.x);
    i=threadIdx.x;
    while(coeff<N/2){  // stop when coeff is half of SIZE (log2(N)-1 iterations) 
        if(i-coeff<0){ // specific case (first equation of the system/subsystem)
            k1=0.0;
            k2=c_i[i]/b_i[i+coeff];
            tb=b_i[i]-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i+coeff]*k2;
            ta=0.0;
            //printf("loop1 check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
        }
        else if(i+coeff>SIZE_CR-1){ // specific case (last equation of the system/subsystem)
            k1=a_i[i]/b_i[i-coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1;
            tc=0.0;
            td=d_i[i]-d_i[i-coeff]*k1;
            //printf("loop2  check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
        }
        else{ // general case
            k1=a_i[i]/b_i[i-coeff];
            k2=c_i[i]/b_i[i+coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i-coeff]*k1-d_i[i+coeff]*k2;
            //printf("loop3  check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
        }
        __syncthreads(); // the first synchronization of threads is to avoid the situation that the other threads use the modified coefficients prepared for the next phase of iteration
        a_i[i]=ta;
        b_i[i]=tb;
        c_i[i]=tc;
        d_i[i]=td;
        __syncthreads(); // the second synchr11onization is to avoid that some threads commence the next phase with vectors unupdated
        //printf("\nIN LOOPIdx: thread : %d block: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,blockIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
        coeff*=2; //augmentation of the amount of stride
    }
    //solution phase
    //printf("loop out\n");
    rhs[2*threadIdx.x+blockIdx.x]=d_i[i]/b_i[i];
}
/*
void test_64_pcr(){
    int D=64;
    //rhs=(float*)malloc(d*sizeof(float));
    float a[64]={0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
    float b[64]={2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};
    float c[64]={-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0};
    float d[64]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    pcr<<<1,64>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}

void test_8_pcr(){
    int D=8;
    //rhs=(float*)malloc(d*sizeof(float));
    float a[8]={0.0,1.0,7.0,2.0,2.0,3.0,-1.0,-1.0};
    float b[8]={1.0,1.0,1.0,11.0,3.0,1.0,2.0,2.0};
    float c[8]={2.0,10.0,2.0,1.0,7.0,2.0,2.0,0.0};
    float d[8]={4.0,14.0,26.0,25.0,0.0,2.0,1.0,3.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    pcr<<<1,8>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));

    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}*/


__global__ void pcr_odd(float *a,float *b,float *c,float *d,float *rhs,int N){
    __shared__ float a_i[SIZE_ODD];    __shared__ float b_i[SIZE_ODD];
    __shared__ float c_i[SIZE_ODD];    __shared__ float d_i[SIZE_ODD];
    int i=threadIdx.x;
    float k1=i==0?0.0:a[i]/b[i-1],k2;
    int M_2_exp=2; // the greatest power of two striclty inferior to N

    while(M_2_exp<N){
        M_2_exp*=2;
    }

    M_2_exp/=2; // this is the criterion of stop for the forward substitution phase, when the stride is larger than half of the SIZE, the loop should break out
    if(i<N-1){
        k2=c[i]/b[i+1];}
    else{k2=0.0;}

    // First reduction
    if(i==0){ // specific case(first equation)
        b_i[i]=b[i]-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i+1]*k2;
        a_i[i]=0.0;
    }
    else if(i==N-1){ // specific case(last equation)
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1;
        c_i[i]=0.0;
        d_i[i]=d[i]-d[i-1]*k1;
    }
    else{ // general case
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    printf("Phase1 check\nI: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
    __syncthreads();
    int coeff=2; //stride beginning from the second phase of forward substitution
    float ta;
    float tb;
    float tc;
    float td;

    //forward substitution loop
    while(coeff<=M_2_exp){
        if(i-coeff<0){ // case where equation to solve dont have a previous equation to eliminate
            k1=0.0;
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            tb=b_i[i]-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i+coeff]*k2;
            ta=0.0;
        }
        else if(i+coeff>N-1){ //last equation of the system or equations which dont have a next equation to eliminate
            k1=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1;
            tc=0.0;
            td=d_i[i]-d_i[i-coeff]*k1;
        }
        else{ // general case
            k1=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i-coeff]*k1-d_i[i+coeff]*k2;
        }
        __syncthreads();
        a_i[i]=ta;
        b_i[i]=tb;
        c_i[i]=tc;
        d_i[i]=td;
        __syncthreads();
        printf("\nIN LOOPIdx: thread : %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
        coeff*=2; //augmentation of the stride

    }
    //solution phase // normally most systems left are equations of 1 unknown since the problem size is not 2^p (p={1,2,3,...}) there will be equations of 3 or 2 unknows left
    //We find these systems not reduced to 1 unknown and find the solutions according to their indexes of the systems
    i=threadIdx.x;
    if( abs(a_i[i])>0.01 || abs(c_i[i])>0.01 ){
        if(i+coeff<N){
            //the first half of the systems,from 0 to N/2-1
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            rhs[i]=(d_i[i]-d_i[i+coeff]*k2)/(b_i[i]-a_i[i+coeff]*k2);
        }
        else if(i-coeff>=0){
            // the second half of the systems, from N/2 to N-1
            k2=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            rhs[i]=(d_i[i]-d_i[i-coeff]*k2)/(b_i[i]-c_i[i-coeff]*k2);
        }
    }
    else{
        // Cases where there is only one unknown left
        rhs[i]=d_i[i]/b_i[i];
    }
    
}

__global__ void pcr_odd2(float *a,float *b,float *c,float *d,float *rhs,int N){
    __shared__ float a_i[ODD2];    __shared__ float b_i[ODD2];
    __shared__ float c_i[ODD2];    __shared__ float d_i[ODD2];
    int i=2*threadIdx.x+blockIdx.x;
    // First reduction (first step)
    float k1=i==0?0.0:a[i]/b[i-1];
    float k2=((i==N-1)||(i==N))?0.0:c[i]/b[i+1];
    int M_2_exp=2; // the greatest power of two striclty inferior to N
    while(M_2_exp<N){
        M_2_exp*=2;
    }
    M_2_exp/=2;
    if(i==N){
        b_i[threadIdx.x]=0.0;
        c_i[threadIdx.x]=0.0;
        d_i[threadIdx.x]=0.0;
        a_i[threadIdx.x]=0.0;
    }
    else if(i==0){ // specific case (first equation)
        b_i[threadIdx.x]=b[i]-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i+1]*k2;
        a_i[threadIdx.x]=0.0;
        
    }
    else if(i==N-1){ // specific case (last equation)
        a_i[threadIdx.x]=-a[i-1]*k1;
        b_i[threadIdx.x]=b[i]-c[i-1]*k1;
        c_i[threadIdx.x]=0.0;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1;
    }
    else{ // general case
        a_i[threadIdx.x]=-a[i-1]*k1;
        b_i[threadIdx.x]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    //printf("Phase1 check1\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
    //printf("Phase1 check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
    __syncthreads();
    int coeff=1; //stride for correspondance between thread and equation
    float ta;
    float tb;
    float tc;
    float td;
    //printf("before loop %d\n",threadIdx.x);
    i=threadIdx.x;
    if(ODD2%2){M_2_exp/=2;}
    while(coeff<M_2_exp){  // stop when coeff is half of SIZE (log2(N)-1 iterations)
        if(2*i+blockIdx.x!=N){
            if(i-coeff<0){ // specific case (first equation of the system/subsystem)
                k1=0.0;
                k2=c_i[i]/b_i[i+coeff];
                tb=b_i[i]-a_i[i+coeff]*k2;
                tc=-c_i[i+coeff]*k2;
                td=d_i[i]-d_i[i+coeff]*k2;
                ta=0.0;
                //printf("loop1 check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
            }
            else if(i+coeff>ODD2-1||(i+coeff==ODD2-1)&&(b_i[ODD2-1]==0.0)){ // specific case (last equation of the system/subsystem)
                k1=a_i[i]/b_i[i-coeff];
                ta=-a_i[i-coeff]*k1;
                tb=b_i[i]-c_i[i-coeff]*k1;
                tc=0.0;
                td=d_i[i]-d_i[i-coeff]*k1;
                //printf("loop2  check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
            }
            else{ // general case
                k1=a_i[i]/b_i[i-coeff];
                k2=c_i[i]/b_i[i+coeff];
                ta=-a_i[i-coeff]*k1;
                tb=b_i[i]-c_i[i-coeff]*k1-a_i[i+coeff]*k2;
                tc=-c_i[i+coeff]*k2;
                td=d_i[i]-d_i[i-coeff]*k1-d_i[i+coeff]*k2;
                //printf("loop3  check\nI: %d threadIdx.x: %d  blockIdx.x: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,threadIdx.x,blockIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
            }
            __syncthreads(); // the first synchronization of threads is to avoid the situation that the other threads use the modified coefficients prepared for the next phase of iteration
            a_i[i]=ta;
            b_i[i]=tb;
            c_i[i]=tc;
            d_i[i]=td;
            __syncthreads(); // the second synchr11onization is to avoid that some threads commence the next phase with vectors unupdated
            //printf("\nIN LOOPIdx: thread : %d block: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,blockIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
        }
        coeff*=2; //augmentation of the amount of stride
    }
    //solution phase
    //printf("loop out\n");
    //printf("\nloop out: thread : %d block: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,blockIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
    if(2*threadIdx.x+blockIdx.x!=N){
        //if(blockIdx.x==1){printf("where???????????????????????,\n");}
        if( abs(a_i[i])>0.00001 || abs(c_i[i])>0.00001 ){
            if(i+coeff<N&&i-coeff<0){
                //the first half of the systems,from 0 to N/2-1
                k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
                rhs[2*i+blockIdx.x]=(d_i[i]-d_i[i+coeff]*k2)/(b_i[i]-a_i[i+coeff]*k2);
            }
            else if(i-coeff>=0){
                // the second half of the systems, from N/2 to N-1
                k2=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
                rhs[2*i+blockIdx.x]=(d_i[i]-d_i[i-coeff]*k2)/(b_i[i]-c_i[i-coeff]*k2);
            }
        }
        else{
            // Cases where there is only one unknown left
            rhs[2*i+blockIdx.x]=d_i[i]/b_i[i];
        }
        //printf("\nsolution phase: thread : %d block: %d sol : %f\n",threadIdx.x,blockIdx.x,rhs[i]);
    }
}
/*
void test_10_pcrodd(){
    int D=10;
    float a[10]={0.0,1.0,7.0,2.0,2.0,3.0,-1.0,2.0,5.0,1.0};
    float b[10]={1.0,1.0,1.0,11.0,3.0,1.0,2.0,1.0,2.0,5.0};
    float c[10]={2.0,10.0,2.0,1.0,7.0,2.0,2.0,1.0,4.0,0.0};
    float d[10]={4.0,14.0,26.0,25.0,0.0,2.0,1.0,3.0,10.0,8.0};
    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    pcr_odd<<<1,10>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    float *ans=(float*)malloc(D*sizeof(float));
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<D;++i){
        printf("X%d = %f\n",i+1,*(ans+i));
    }
    free(ans);
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
}*/

/*
void test_512_all(){
    int D=512; //dimensions of our problem
    //rhs=(float*)malloc(d*sizeof(float));
    float a[512];
    float b[512];
    float c[512];
    float d[512];
    a[0]=0.0;   a[511]=-1.0;
    b[0]=2.0;   b[511]=2.0;
    c[0]=-1.0;  c[511]=0.0;
    d[0]=1.0;   d[511]=1.0;
    for(int i=1;i<511;++i){
        a[i]=-1.0;
        b[i]=2.0;
        c[i]=-1.0;
        d[i]=1.0;
    }


    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    

    float temps_ecoule=0.0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	cr<<<1,SIZE_CR>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps_ecoule, start, stop);
    float *ans=(float*)malloc(D*sizeof(float));
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    /*
    // all this part is to verify the solution of CR algorithm
    printf("CR algorithm answer:\n\n");
    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }
    

    printf("\nRun time elapsed on GPU, Cyclic Reduction: %f millisecondes\n", temps_ecoule);
    temps_ecoule=0.0;
    
    cudaMemcpy(rhs,ans,D*sizeof(float),cudaMemcpyHostToDevice); // reinitialise all the solutions values to zero
    cudaEventRecord(start, 0);
	pcr<<<2,SIZE_CR>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&temps_ecoule, start, stop);
    printf("\nRun time elapsed on GPU, Parallel Cyclic Reduction: %f millisecondes\n", temps_ecoule);

    
    // all this part is to verify the solution of pcr algorithm
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    
    printf("PCR algorithm answer:\n\n");
    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }
    

    auto t_start = std::chrono::high_resolution_clock::now();
    thomas(a,b,c,d,ans,D);
    auto t_end = std::chrono::high_resolution_clock::now();
    temps_ecoule = std::chrono::duration<float, std::milli>(t_end-t_start).count();
    printf("\nRun time elapsed on CPU, Thomas Method: %f millisecondes\n", temps_ecoule);
    //printf("\nThomas algorithm run time : %g milliseconds\n",elapsed_time_ms);

    /* //verify thomas algorithm answer
    printf("Thomas algorithm answer:\n\n");    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }

    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
    D=511;
    float a_[511];
    float b_[511];
    float c_[511];
    float d_[511];
    a_[0]=0.0;   a_[510]=-1.0;
    b_[0]=2.0;   b_[510]=2.0;
    c_[0]=-1.0;  c_[510]=0.0;
    d_[0]=1.0;   d_[510]=1.0;
    for(int i=1;i<510;++i){
        a_[i]=-1.0;
        b_[i]=2.0;
        c_[i]=-1.0;
        d_[i]=1.0;
    }
    float *a_g,*b_g,*c_g,*d_g;
    float *rhs_;
    cudaMalloc(&rhs_,D*sizeof(float));
    cudaMalloc(&a_g,D*sizeof(float));
    cudaMalloc(&b_g,D*sizeof(float));
    cudaMalloc(&c_g,D*sizeof(float));
    cudaMalloc(&d_g,D*sizeof(float));
    cudaMemcpy(a_g,a_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_g,b_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_g,c_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_g,d_,D*sizeof(float),cudaMemcpyHostToDevice);

    temps_ecoule=0.0;
    cudaEventRecord(start, 0);
	pcr_odd<<<1,SIZE_ODD>>>(a_g,b_g,c_g,d_g,rhs_,D);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps_ecoule, start, stop);
    printf("\nRun time elapsed on GPU, Algorithm Parallel Reduction Cyclique With ODD SIZE: %f millisecondes\n", temps_ecoule);

    /* //verifiy pcr_odd answer
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    printf("PCR algorithm answer:\n\n");
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }
    
    free(ans);
    cudaFree(a_g);    cudaFree(b_g);    cudaFree(c_g);    cudaFree(d_g);
    cudaFree(rhs_);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
*/
void test_128_pcr(){
    int D=128; //dimensions of our problem
    //rhs=(float*)malloc(d*sizeof(float));
    float a[128];
    float b[128];
    float c[128];
    float d[128];
    a[0]=0.0;   a[127]=-1.0;
    b[0]=2.0;   b[127]=2.0;
    c[0]=-1.0;  c[127]=0.0;
    d[0]=1.0;   d[127]=1.0;
    for(int i=1;i<127;++i){
        a[i]=-1.0;
        b[i]=2.0;
        c[i]=-1.0;
        d[i]=1.0;
    }


    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    

    float temps_ecoule=0.0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    pcr<<<2,SIZE_CR>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps_ecoule, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\nRun time elapsed on GPU, Parallel Cyclic Reduction: %f millisecondes\n", temps_ecoule);
    float *ans=(float*)malloc(D*sizeof(float));
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%10==0){
            printf("\n");
        }
    }
    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
    free(ans);
}

void test_1024_all(){
    int D=1024; //dimensions of our problem
    //rhs=(float*)malloc(d*sizeof(float));
    float a[1024];
    float b[1024];
    float c[1024];
    float d[1024];
    a[0]=0.0;   a[1023]=-1.0;
    b[0]=2.0;   b[1023]=2.0;
    c[0]=-1.0;  c[1023]=0.0;
    d[0]=1.0;   d[1023]=1.0;
    for(int i=1;i<1023;++i){
        a[i]=-1.0;
        b[i]=2.0;
        c[i]=-1.0;
        d[i]=1.0;
    }


    float *a_gpu,*b_gpu,*c_gpu,*d_gpu;
    float *rhs;
    cudaMalloc(&rhs,D*sizeof(float));
    cudaMalloc(&a_gpu,D*sizeof(float));
    cudaMalloc(&b_gpu,D*sizeof(float));
    cudaMalloc(&c_gpu,D*sizeof(float));
    cudaMalloc(&d_gpu,D*sizeof(float));
    cudaMemcpy(a_gpu,a,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu,c,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu,d,D*sizeof(float),cudaMemcpyHostToDevice);
    

    float temps_ecoule=0.0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	cr<<<1,SIZE_CR>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps_ecoule, start, stop);
    float *ans=(float*)malloc(D*sizeof(float));
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }
    /*
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    
    // all this part is to verify the solution of CR algorithm
    printf("CR algorithm answer:\n\n");
    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }
    */

    printf("\nRun time elapsed on GPU, Cyclic Reduction: %f microsecondes\n", 1000*temps_ecoule);
    temps_ecoule=0.0;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(rhs,ans,D*sizeof(float),cudaMemcpyHostToDevice); // reinitialise all the solutions values to zero
    
    auto t_start = std::chrono::high_resolution_clock::now();
    pcr<<<2,SIZE_CR>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
    //cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    temps_ecoule = std::chrono::duration<float, std::micro>(t_end-t_start).count();
    printf("\nRun time elapsed on GPU, Parallel Cyclic Reduction: %f microsecondes\n", temps_ecoule);
    
    /*
    // all this part is to verify the solution of pcr algorithm
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    
    printf("PCR algorithm answer:\n\n");
    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }
    for(int i=0;i<D;++i){
        ans[i]=0.0;
    }*/
    
    

    t_start = std::chrono::high_resolution_clock::now();
    thomas(a,b,c,d,ans,D);
    t_end = std::chrono::high_resolution_clock::now();
    temps_ecoule = std::chrono::duration<float, std::micro>(t_end-t_start).count();
    printf("\nRun time elapsed on CPU, Thomas Method: %f microsecondes\n", temps_ecoule);
    //printf("\nThomas algorithm run time : %g milliseconds\n",elapsed_time_ms);

    /* //verify thomas algorithm answer
    printf("Thomas algorithm answer:\n\n");    
    for(int i=0;i<D;++i){
        printf("X%d = %f ",i+1,*(ans+i));
        if(i>0&&i%24==0){
            printf("\n");
        }
    }*/

    cudaFree(a_gpu);    cudaFree(b_gpu);    cudaFree(c_gpu);    cudaFree(d_gpu);
    cudaFree(rhs);
    D=1023;
    float a_[1023];
    float b_[1023];
    float c_[1023];
    float d_[1023];
    a_[0]=0.0;   a_[1022]=-1.0;
    b_[0]=2.0;   b_[1022]=2.0;
    c_[0]=-1.0;  c_[1022]=0.0;
    d_[0]=1.0;   d_[1022]=1.0;
    for(int i=1;i<1022;++i){
        a_[i]=-1.0;
        b_[i]=2.0;
        c_[i]=-1.0;
        d_[i]=1.0;
    }
    float *a_g,*b_g,*c_g,*d_g;
    float *rhs_;
    float *a2=(float*)malloc(D*sizeof(float));
    cudaMalloc(&rhs_,D*sizeof(float));
    cudaMalloc(&a_g,D*sizeof(float));
    cudaMalloc(&b_g,D*sizeof(float));
    cudaMalloc(&c_g,D*sizeof(float));
    cudaMalloc(&d_g,D*sizeof(float));
    cudaMemcpy(a_g,a_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_g,b_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(c_g,c_,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_g,d_,D*sizeof(float),cudaMemcpyHostToDevice);
    temps_ecoule=0.0;
    t_start = std::chrono::high_resolution_clock::now();
    pcr_odd2<<<2,ODD2>>>(a_g,b_g,c_g,d_g,rhs_,D);
    t_end = std::chrono::high_resolution_clock::now();
    temps_ecoule = std::chrono::duration<float, std::micro>(t_end-t_start).count();
    printf("\nRun time elapsed on GPU, Algorithm Parallel Reduction Cyclique With ODD SIZE: %f microsecondes\n", temps_ecoule);
    /*
     //verifiy pcr_odd answer
    cudaMemcpy(ans,rhs,D*sizeof(float),cudaMemcpyDeviceToHost);
    thomas(a_,b_,c_,d_,a2,D);
    
    printf("PCR ODD algorithm answer:\n\n");
    for(int i=0;i<D;++i){
        printf("X%d = %f %f ",i+1,*(ans+i),a2[i]);
        if(i>0&&i%4==0){
            printf("\n");
        }
    }*/
    
    free(ans);  free(a2);
    cudaFree(a_g);    cudaFree(b_g);    cudaFree(c_g);    cudaFree(d_g);
    cudaFree(rhs_);
}

int main(){
    /*
    int N=10;
    float a[10]={0.0,1.0,7.0,2.0,2.0,3.0,-1.0,2.0,5.0,1.0};
    float b[10]={1.0,1.0,1.0,11.0,3.0,1.0,2.0,1.0,2.0,5.0};
    float c[10]={2.0,10.0,2.0,1.0,7.0,2.0,2.0,1.0,4.0,0.0};
    float d[10]={4.0,14.0,26.0,25.0,0.0,2.0,1.0,3.0,10.0,8.0};
    float ref[10]={-1.9454,2.9727,1.2973,1.9469,0.9894,-0.9803,0.0060,0.0038,2.9841,1.0032};
    float *rhs;
    rhs=(float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        rhs[i]=0.0;
    }
    thomas(a,b,c,d,rhs,N);
    for(int i=0;i<N;++i){
        printf("%d th standard answer : %f    our solution: %f\n",i+1,ref[i],rhs[i]);
    }
    free(rhs);*/

    /*
    test_8_cr();
    printf("check err8\n")*/
    
    /*
    test_64_cr();
    printf("check err64\n");*/
/*
    test_64_pcr();
    printf("normally out pcr64\n");*/

    test_1024_all();
    return 0;
}

/*
    float a[8]={0.0,1.0,7.0,2.0,2.0,3.0,-1.0,-1.0}
    float b[8]={1.0,1.0,1.0,11.0,3.0,1.0,2.0,2.0};
    float c[8]={2.0,10.0,2.0,1.0,7.0,2.0,2.0,0.0};
    float d[8]={4.0,14.0,26.0,25.0,0.0,2.0,1.0,3.0};
    float rhs[8]={-1.9896,2.9948,1.2995,1.8685,1.8476,-1.3257,-1.1086,0.9457};
*/


/*sol for 16*16 system
8.0000,15.0000,21.0000,26.0000,30.0000,33.0000,35.0000,36.0000,36.0000,35.0000,33.0000,30.0000,26.0000,21.0000,15.0000,8.0000*/


/*sol for 32*32 system
16.0000,31.0000,45.0000,58.0000,70.0000,81.0000, 91.0000,100.0000,         108.0000,115.0000,121.0000,126.0000,130.0000,133.0000,135.0000,136.0000,
136.0000,135.0000,133.0000,130.0000,126.0000,121.0000,115.0000,108.0000,   100.0000,91.0000,81.0000,70.0000,58.0000,45.0000,31.0000,16.0000

64*64:

32.0000 63.0000 93.0000 122.0000 150.0000 177.0000 203.0000 228.0000 252.0000 275.0000 297.0000 318.0000 338.0000 357.0000 375.0000 392.0000 408.0000 423.0000
 437.0000 450.0000 462.0000 473.0000 483.0000 492.0000 500.0000 507.0000 513.0000 518.0000 522.0000 525.0000 527.0000 528.0000 528.0000 527.0000
  525.0000
  522.0000
  518.0000
  513.0000
  507.0000
  500.0000
  492.0000
  483.0000
  473.0000
  462.0000
  450.0000
  437.0000
  423.0000
  408.0000
  392.0000
  375.0000
  357.0000
  338.0000
  318.0000
  297.0000
  275.0000
  252.0000
  228.0000
  203.0000
  177.0000 150.0000 122.0000 93.0000 63.0000 32.0000

  
  */