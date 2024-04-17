#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include "bmp_utility.h"

#define HW_REGS_BASE (0xff200000)
#define HW_REGS_SPAN (0x00200000)
#define HW_REGS_MASK (HW_REGS_SPAN - 1)
#define LED_BASE 0x1000
#define PUSH_BASE 0x3010
#define VIDEO_BASE 0x0000

#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240

#define FPGA_ONCHIP_BASE     (0xC8000000)
#define IMAGE_SPAN (IMAGE_WIDTH * IMAGE_HEIGHT * 4)
#define IMAGE_MASK (IMAGE_SPAN - 1)




int main(void) {
    volatile unsigned int *video_in_dma = NULL;
    volatile unsigned int *key_ptr = NULL;
    volatile unsigned short *video_mem = NULL;
    void *virtual_base, *video_base;
    int fd;

    // Open /dev/mem
    if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1) {
        printf("ERROR: could not open \"/dev/mem\"...\n");
        return 1;
    }

    // Map physical memory into virtual address space
    virtual_base = mmap(NULL, HW_REGS_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, HW_REGS_BASE);
    if (virtual_base == MAP_FAILED) {
        printf("ERROR: mmap() failed...\n");
        close(fd);
        return 1;
    }
    video_base = mmap(NULL, IMAGE_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, FPGA_ONCHIP_BASE);

    // Calculate the virtual address where our device is mapped
    video_in_dma = (unsigned int *)(virtual_base + ((VIDEO_BASE) & (HW_REGS_MASK)));
    key_ptr = (unsigned int*)(virtual_base + ((PUSH_BASE) & (HW_REGS_MASK)));
    video_mem = (unsigned short*)(video_base + ((FPGA_ONCHIP_BASE) & (IMAGE_MASK)));
   

    int value = *(video_in_dma+3);

    printf("Video In DMA register updated at:0%x\n",(video_in_dma));

    // Modify the PIO register
    *(video_in_dma+3) = 0x4;
    //*h2p_lw_led_addr = *h2p_lw_led_addr + 1;

    value = *(video_in_dma+3);

    printf("enabled video:0x%x\n",value);
    

    //change 
    const char* filename = "final_image_color.bmp";

    while(1){
	if(*key_ptr != 0x07){
		printf("Button pressed\n");
		while(*key_ptr != 0x07){}
		*(video_in_dma+3) = 0x0;
		break;
	}
}

   unsigned short pixels[IMAGE_HEIGHT][IMAGE_WIDTH];
   int i,j;
   for (i = 0; i < IMAGE_HEIGHT; i++){
	for (j = 0; j < IMAGE_WIDTH; j++){
		pixels[i][j] = *(video_mem + (i<<9) + j);
	}
}

    // Saving image as color
    saveImageShort(filename,&pixels[0][0],IMAGE_WIDTH,IMAGE_HEIGHT);

for (i = 0; i<IMAGE_HEIGHT; i++){
	for (j = 0; j < IMAGE_WIDTH; j++){
		unsigned short temp = *(video_mem + (i<<9) + j);
		if((((int)((temp & 0b1111100000000000)>>10) + ((int)((temp & 0b0000011111100000)>>5))/2 + (int)(temp & 0b0000000000011111)))/3 > 12){
			temp = 0xffff;
		}
		else{
			temp = 0x0000;
		}
		pixels[i][j] = temp;
	}	
}

    const char* filename1 = "final_image_bw.bmp";
    //saving image as black and white
    saveImageShort(filename1,&pixels[0][0],320,240);


    // Clean up
    if (munmap(virtual_base, IMAGE_SPAN) != 0) {
        printf("ERROR: munmap() failed...\n");
        close(fd);
        return 1;
    }



    close(fd);
    return 0;
}
