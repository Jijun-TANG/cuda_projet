#include<stdio.h>
#include<iostream>
#define ind 32
#define SIZE 64
__global__ void pcr_odd(float *a,float *b,float *c,float *d,float *rhs,int N);

__host__ void thomas(float *a,float *b,float *c,float *d,float *rhs,int N){
    c[0]=c[0]/b[0];
    d[0]=d[0]/b[0];
    b[0]=1.0;
    for(int i=1;i<N;++i){
        float t1=b[i]-c[i-1]*a[i];
        printf("%d ieme denominator val1 %.1f\n",i,t1);
        if(abs(t1)<0.00001){
            c[i]=0.0;
            d[i]=d[i]-d[i-1]*a[i];
        }
        else{
            c[i]=c[i]/(t1);
            d[i]=(d[i]-d[i-1]*a[i])/(t1);
        }
        std::cout<<i<<"th iteration   c[i]= "<<c[i]<<" d[i]= "<<d[i]<<'\n';
    }
    rhs[N-1]=d[N-1];
    for(int i=N-2;i>=0;--i){
        rhs[i]=d[i]-c[i]*rhs[i+1];
    }
}

__global__ void cr(float *a,float *b,float *c,float *d,float *rhs,int N){
    //int ind=N/2;
    __shared__ float a_i[ind];    __shared__ float b_i[ind];
    __shared__ float c_i[ind];    __shared__ float d_i[ind];
    int coeff=2;
    int i=(threadIdx.x+1)*coeff-1;
    float k1=a[i]/b[i-1],k2; ///
    if(i<N-1){
        k2=c[i]/b[i+1];}  /// calculate k1 k2
    else{k2=0.0;}      ///
    a_i[threadIdx.x]=-a[i-1]*k1;
    if(i!=N-1){
        b_i[threadIdx.x]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[threadIdx.x]=-c[i+1]*k2;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    else{
        b_i[threadIdx.x]=b[i]-c[i-1]*k1;
        c_i[threadIdx.x]=0.0;
        d_i[threadIdx.x]=d[i]-d[i-1]*k1;
    }
    printf("Idx: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",threadIdx.x,k1,k2,a_i[threadIdx.x],b_i[threadIdx.x],c_i[threadIdx.x],d_i[threadIdx.x]);
    __syncthreads();
    int num=ind/2;        //maximum number of threads needed

    //forward substitution
    do{
        i=(threadIdx.x+1)*coeff-1;
        if(threadIdx.x<num){
            k1=a_i[i]/b_i[i-coeff/2];
            a_i[i]=-a_i[i-coeff/2]*k1;
            if(threadIdx.x!=num-1){
                k2=c_i[i]/b_i[i+coeff/2];
                b_i[i]=b_i[i]-c_i[i-coeff/2]*k1-a_i[i+coeff/2]*k2;
                c_i[i]=-c_i[i+coeff/2]*k2;
                d_i[i]=d_i[i]-d_i[i-coeff/2]*k1-d_i[i+coeff/2]*k2;
            }
            else{
                b_i[i]=b_i[i]-c_i[i-coeff/2]*k1;
                c_i[i]=0.0;
                d_i[i]=d_i[i]-d_i[i-coeff/2]*k1;
            }
            printf("\nIN LOOPIdx: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n",threadIdx.x,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
        }
        num=num/2;
        coeff=coeff*2;
    }while(num>1);
    //solve the 2 unknowns Xn/2 and Xn and the other 2 between (0,Xm/2) & (Xm/2,Xm)
    if(threadIdx.x==0){
        if(b_i[i]==0){
            rhs[4*i+3]=d_i[i]/c_i[i];
            rhs[2*i+1]=(d_i[2*i+1]-b_i[2*i+1]*rhs[4*i+3])/a_i[2*i+1];
            printf("condition 1\n");
        }
        else if(a_i[2*i+1]==0){
            rhs[4*i+3]=d_i[2*i+1]/b_i[2*i+1];
            rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[4*i+3])/b_i[2*i+1];
            printf("condition 2\n");
        }
        else{
            rhs[4*i+3]=(d_i[i]-d_i[2*i+1]*b_i[i]/a_i[2*i+1])/(c_i[i]-b_i[2*i+1]*b_i[i]/a_i[2*i+1]);
            rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[4*i+3])/b_i[i];
            printf("condition 3\n");
        }
        printf("first result check %d %d\n%f\n%f\n",2*i+1,4*i+3,rhs[2*i+1],rhs[4*i+3]);

        // solve the precedente matrix
        if(N>4){
            rhs[3*i+2]=(d_i[(3*i+1)/2]-a_i[(3*i+1)/2]*rhs[2*i+1]-c_i[(3*i+1)/2]*rhs[4*i+3])/b_i[(3*i+1)/2];
            rhs[i]=(d_i[i/2]-c_i[i/2]*rhs[2*i+1])/b_i[i/2];
        }
        printf("second result check %d %d\n%f\n%f\n",i,3*i+2,rhs[i],rhs[3*i+2]);
    }
    ///check the first 4 solutions of system
    /*
    for(int j=0;j<N;++j){
        printf("111X%d = %f\n",j+1,*(rhs+j));
    }*/
    

    //backward substitution
    num=num*4;
    coeff/=4;
    if(threadIdx.x==0){
        printf("maximum num threads value %d coeff value %d\n",num,coeff);
    }
    while(num<ind){
        i=-1+(coeff/2)+threadIdx.x*coeff;
        if(threadIdx.x<num){
            if(threadIdx.x==0){
                rhs[2*i+1]=(d_i[i]-c_i[i]*rhs[2*i+1+coeff])/b_i[i];
            }
            else{
                rhs[2*i+1]=(d_i[i]-a_i[i]*rhs[2*i+1-coeff]-c_i[i]*rhs[2*i+1+coeff])/b_i[i];
            }
            printf("backward in loop value rhs[%d] = %f\n",2*i+1,rhs[2*i+1]);
        }
        __syncthreads();
        num*=2;
        coeff/=2;
    }
    ///check backward propogation
    printf("check backward propogation rhs[%d] = %f\nrhs[%d] = %f\n",threadIdx.x,*(rhs+threadIdx.x),threadIdx.x+1,*(rhs+threadIdx.x+1));
    if(threadIdx.x<ind){
        i=2*threadIdx.x;
        if(i==0){
            rhs[i]=(d[i]-c[i]*rhs[i+1])/b[i];
        }
        else{
            rhs[i]=(d[i]-c[i]*rhs[i+1]-a[i]*rhs[i-1])/b[i];
        }
    }
}

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
}

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
    cr<<<1,32>>>(a_gpu,b_gpu,c_gpu,d_gpu,rhs,D);
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
    //int ind=N/2;
    __shared__ float a_i[SIZE];    __shared__ float b_i[SIZE];
    __shared__ float c_i[SIZE];    __shared__ float d_i[SIZE];
    int i=threadIdx.x;
    float k1=i==0?0.0:a[i]/b[i-1],k2; ///
    if(i<N-1){
        k2=c[i]/b[i+1];}  /// calculate k1 k2
    else{k2=0.0;}      ///


    if(i==0){
        b_i[i]=b[i]-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i+1]*k2;
        a_i[i]=0.0;
    }
    else if(i==N-1){
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1;
        c_i[i]=0.0;
        d_i[i]=d[i]-d[i-1]*k1;
    }
    else{
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    printf("Phase1 check\nI: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
    __syncthreads();
    int coeff=2;
    float ta;
    float tb;
    float tc;
    float td;
    while(coeff<=SIZE/2){  /// l'algo stop when coeff is half of SIZE

        if(i==0||i-coeff<0){
            k1=0.0;
            k2=c_i[i]/b_i[i+coeff];
            tb=b_i[i]-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i+coeff]*k2;
            ta=0.0;
        }
        else if(i==N-1||i+coeff>N-1){
            k1=a_i[i]/b_i[i-coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1;
            tc=0.0;
            td=d_i[i]-d_i[i-coeff]*k1;
        }
        else{
            k1=a_i[i]/b_i[i-coeff];
            k2=c_i[i]/b_i[i+coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i-coeff]*k1-d_i[i+coeff]*k2;
        }
        __syncthreads();
        printf("\nloop result i: %d a: %f  b: %f  c: %f  d: %f\n",i,ta,tb,tc,td);
        a_i[i]=ta;
        b_i[i]=tb;
        c_i[i]=tc;
        d_i[i]=td;
        __syncthreads();
        coeff*=2;
    }
    //solution phase
    rhs[i]=d_i[i]/b_i[i];
}

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
}


__global__ void pcr_odd(float *a,float *b,float *c,float *d,float *rhs,int N){
    __shared__ float a_i[SIZE];    __shared__ float b_i[SIZE];
    __shared__ float c_i[SIZE];    __shared__ float d_i[SIZE];
    int i=threadIdx.x;
    float k1=i==0?0.0:a[i]/b[i-1],k2; ///
    int M_2_exp=2;
    while(M_2_exp<N){
        M_2_exp*=2;
    }
    M_2_exp/=4;
    if(i<N-1){
        k2=c[i]/b[i+1];}  /// calculate k1 k2
    else{k2=0.0;}      ///
    if(i==0){
        b_i[i]=b[i]-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i+1]*k2;
        a_i[i]=0.0;
    }
    else if(i==N-1){
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1;
        c_i[i]=0.0;
        d_i[i]=d[i]-d[i-1]*k1;
    }
    else{
        a_i[i]=-a[i-1]*k1;
        b_i[i]=b[i]-c[i-1]*k1-a[i+1]*k2;
        c_i[i]=-c[i+1]*k2;
        d_i[i]=d[i]-d[i-1]*k1-d[i+1]*k2;
    }
    printf("Phase1 check\nI: %d k1: %f  k2: %f\n a[i]: %f   b[i] : %f   c[i] : %f   d[i] : %f   \n\n",i,k1,k2,a_i[i],b_i[i],c_i[i],d_i[i]);
    __syncthreads();
    int coeff=2;
    float ta;
    float tb;
    float tc;
    float td;
    while(coeff<=M_2_exp){
        if(i==0||i-coeff<0){
            k1=0.0;
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            tb=b_i[i]-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i+coeff]*k2;
            ta=0.0;
        }
        else if(i==N-1||i+coeff>N-1){
            k1=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1;
            tc=0.0;
            td=d_i[i]-d_i[i-coeff]*k1;
        }
        else{
            k1=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            ta=-a_i[i-coeff]*k1;
            tb=b_i[i]-c_i[i-coeff]*k1-a_i[i+coeff]*k2;
            tc=-c_i[i+coeff]*k2;
            td=d_i[i]-d_i[i-coeff]*k1-d_i[i+coeff]*k2;
        }
        __syncthreads();
        printf("\nloop result i: %d a: %f  b: %f  c: %f d: %f  k1 : %f  k2 : %f\n",i,ta,tb,tc,td,k1,k2);
        a_i[i]=ta;
        b_i[i]=tb;
        c_i[i]=tc;
        d_i[i]=td;
        __syncthreads();
        coeff*=2;
    }
    //solution phase
    i=threadIdx.x;
    //printf("check re: i : %d\n a: %f  b: %f  c: %f d: %f  k1 : %f  k2 : %f\n",i,a_i[i],b_i[i],c_i[i],d_i[i],k1,k2);
    if( abs(a_i[i])>0.01 || abs(c_i[i])>0.01 ){
        printf("why ur in?:  i : %d\n a: %f  b: %f  c: %f d: %f  k1 : %f  k2 : %f\n",i,a_i[i],b_i[i],c_i[i],d_i[i],k1,k2);
        if(i+coeff<N){
            k2=b_i[i+coeff]==0.0?0.0:c_i[i]/b_i[i+coeff];
            rhs[i]=(d_i[i]-d_i[i+coeff]*k2)/(b_i[i]-a_i[i+coeff]*k2);
            //rhs[i+coeff]=(d_i[i+coeff]-a_i[i+coeff]*rhs[i])/b_i[i+coeff];
            printf("check result: i: %d\ncoeff: %d :%f %f\n",i,coeff,rhs[i],rhs[i+coeff]);
        }
        else if(i-coeff>=0){
            k2=b_i[i-coeff]==0.0?0.0:a_i[i]/b_i[i-coeff];
            rhs[i]=(d_i[i]-d_i[i-coeff]*k2)/(b_i[i]-c_i[i-coeff]*k2);
        }
    }
    else{
        printf("instructions in\n");
        rhs[i]=d_i[i]/b_i[i];
    }
}

void test_10_pcrodd(){
    int D=10;
    //rhs=(float*)malloc(d*sizeof(float));
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
    printf("check err64\n");
    */
    
    test_64_pcr();
    printf("normally out pcr64\n");
    
    /*
    test_10_pcrodd();
    printf("all out\n");*/
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
  177.0000 150.0000 122.0000 93.0000 63.0000 32.0000*/