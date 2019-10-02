#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h> 
#include <sys/socket.h> 
#include <sys/types.h>
#include <strings.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>

// under Linux 4.18 tcp pacing through socket is misbehaving and paces slow_start
#define SUPPORT_PACING 0
// size of buffer for sending and receiving : here 128 kB
#define BUFSIZE (128 * 1024)


// buffer
char buf[BUFSIZE];

/**
 * server is receiver, so just receive all the data to keep receiver window open
 * arg: socket fd (ptr used to represent int ... )
 **/
void* server_session(void* arg) {
    int s = (int)((intptr_t) arg);
    for(;;) {
        int received = recv(s, buf, BUFSIZE, 0);
        if(received <= 0) {
            break;
        }
    }
}

/**
 * server accepts all connections
 * arg: listen socket fd (ptr used to represent int ... )
 **/
void* server_accept(void* arg) {
    int listens = (int)((intptr_t) arg);
    for(;;) {
        int s = accept(listens, NULL, NULL);
        if(s < 0) {
            perror("accept failed");
            exit(1);
        }
        // new thread to receive data
        pthread_t thread;
        pthread_create(&thread, NULL, server_session, (void*)((intptr_t) s));
    }
}

/**
 * client sends data non-stop
 * arg: socket fd (ptr used to represent int ... )
 **/
void* client_send(void* arg) {
    int s = (int)((intptr_t) arg);
    for(;;) {
        if (send(s, buf, BUFSIZE, 0) == -1){
            perror("send failed");
            exit (1);
        }
    }
}

/**
 * trafficgen main func
 * argc: number of arguments
 * argv: arguments
 **/
int main(int argc, char *argv[]) {
    bool pacing = false;
    int seconds = -1;
    char *cong  = "cubic";
    int port    = 5001;
    bool server = false;
    int opt;
    char *host  = "localhost";

    // parse arguments

    while ((opt = getopt(argc, argv, "h:sPp:t:c:")) != -1) {
        switch (opt) {
        case 'h': host    = optarg; break;
        case 'P': pacing  = true; break;
        case 'p': port    = (int) strtol(optarg, (char **)NULL, 10); break;
        case 's': server  = true; break;
        case 't': seconds = (int) strtol(optarg, (char **)NULL, 10); break;
        case 'c': cong    = optarg; break;
        default:
            fprintf(stderr, "Usage: %s     [-s] [-P] [-h host] [-p port] [-t seconds] [-c cong]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if(server) {
        // is server mode
        struct sockaddr_in servaddr;
        struct epoll_event event;
        time_t start_t, now_t;
        bzero(&servaddr, sizeof(servaddr));

        // assign IP, PORT and listen
        servaddr.sin_family = AF_INET; 
        servaddr.sin_addr.s_addr = htonl(INADDR_ANY); 
        servaddr.sin_port = htons(port);
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if(sockfd < 0) {
            perror("socket creation failed");
            exit(1);
        }
        int reuse_addr = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr)) < 0) {
            perror("setsockopt reuseaddr failed"); 
            close(sockfd);
            exit(1); 
        }
        if (bind(sockfd, (struct sockaddr*) &servaddr, sizeof(servaddr))) { 
            perror("bind failed"); 
            close(sockfd);
            exit(1); 
        }
        if(listen(sockfd, INT_MAX)) {
            perror("listen failed");
            close(sockfd);
            exit(1);
        }

        // create thread for accept to concurrently keep time
        pthread_t thread;
        pthread_create(&thread, NULL, server_accept, (void*)((intptr_t) sockfd));
        time(&start_t);

        // time keeping
        for(;;) {
            sleep(1);
            time(&now_t);
            if(difftime(now_t, start_t) > seconds && seconds > 0) {
                fprintf(stderr, "time is over\n");
                break;
            }
        }
        close(sockfd);
    }
    else {
        // client mode sending the data

        // initialize buffer to 012345678901234567890..
        for(int i = 0; i < BUFSIZE; i++) {
            buf[i] = '0' + (i % 10);
        }

        time_t start_t, now_t;
        struct sockaddr_in servaddr;
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);

        // connect to server
        if(sockfd < 0) {
            perror("socket creation failed");
            exit(1);
        }
        struct hostent *he;
        if ((he=gethostbyname(host)) == NULL) {
            perror("gethostbyname failed");
            close(sockfd);
            exit(1);
        }
        bzero(&servaddr, sizeof(servaddr)); 
        servaddr.sin_family = AF_INET;
        servaddr.sin_port = htons(port);
        servaddr.sin_addr = *((struct in_addr *)he->h_addr);
        fprintf(stderr, "set congestion control: %s\n",cong);

        // manually adapt congestion control variant
        if (setsockopt(sockfd, IPPROTO_TCP, TCP_CONGESTION, cong, strlen(cong)) < 0) {
            perror("setsockopt cong failed");
            close(sockfd);
            exit(1);
        }

        //whether we should do pacing (or done via tc)
        if(pacing) {
#if SUPPORT_PACING
            fprintf(stderr, "set pacing\n");
            unsigned int rate = INT_MAX;
            if (setsockopt(sockfd, SOL_SOCKET, SO_MAX_PACING_RATE, &rate, sizeof(rate)) < 0) {
                perror("setsockopt pacing failed");
                close(sockfd);
                exit(1);
            }
#else
            fprintf(stderr, "pacing unsupported\n");
#endif
        }
        if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(struct sockaddr)) == -1) {
            perror("connect failed");
            close(sockfd);
            exit(1);
        }

        //start send thread to keep time concurrently
        pthread_t thread;
        pthread_create(&thread, NULL, client_send, (void*)((intptr_t) sockfd));
        time(&start_t);

        //time keeping
        for(;;) {
            sleep(1);
            time(&now_t);
            if(difftime(now_t, start_t) > seconds && seconds > 0) {
                fprintf(stderr, "time is over\n");
                break;
            }
        }
        close(sockfd);
    }
}
