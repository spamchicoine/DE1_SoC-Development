// #include "address_map_arm.h"
#include <time.h>


#define KEY_BASE              0xFF200050
#define VIDEO_IN_BASE         0xFF203060
#define FPGA_ONCHIP_BASE      0xC8000000
#define FPGA_BUFFER_BASE	0x08000000
#define CHAR_BUFFER_BASE	0xC9000000

/* This program demonstrates the use of the D5M camera with the DE1-SoC Board
 * It performs the following: 
 * 	1. Capture one frame of video when any key is pressed.
 * 	2. Display the captured frame when any key is pressed.		  
*/
/* Note: Set the switches SW1 and SW2 to high and rest of the switches to low for correct exposure timing while compiling and the loading the program in the Altera Monitor program.
*/
void timestamp(volatile short * Char_Mem_ptr){
	// Get time value
	time_t rawtime;
  	struct tm * timeStamp;

	time (&rawtime);
	timeStamp = localtime (&rawtime);
	//char buf[10];
	//sprintf(buf,"%s", asctime(timeStamp));
	int x = 5;
	int y = 1;
	int offset = (y<<7) + x;
	char * text_ptr = asctime(timeStamp);

	while (*(text_ptr)){
		*((Char_Mem_ptr + offset)) = *(text_ptr);
		
		++text_ptr;
		++offset;
	}
}


void standard(volatile int *KEY_ptr, volatile int * Video_In_DMA_ptr){
	while (1)
		{
			if (*KEY_ptr == 1)						// check if any KEY was pressed
			{
				*(Video_In_DMA_ptr + 3)	= 0x4;
				while (*KEY_ptr != 0);
				break;
			}
		}
}

void blackwhite(volatile short *Video_Mem_ptr){
	int x, y;
	for (y = 0; y < 240; y++) {
		for (x = 0; x < 320; x++) {
			short temp2 = *(Video_Mem_ptr + (y << 9) + x);

			if ((((int)((temp2 & 0b1111100000000000) >> 10) + ((int)((temp2 & 0b0000011111100000) >> 5))/2 + (int)((temp2 & 0b0000000000011111)))/3) > 12){
				temp2 = 0xffff;
			}
			else{
				temp2 = 0x0000;
			}

			*(Video_Mem_ptr + (y << 9) + x) = temp2;
		}
	}
}

void flip(volatile short *Video_Mem_ptr){
	int x,y;
	for (y = 0; y < 120; y++) {
		for (x = 0; x < 320; x++) {
			short temp2 = *(Video_Mem_ptr + (y << 9) + x);
			short temp3 = *(Video_Mem_ptr + ((240-y) << 9) + x);

			*(Video_Mem_ptr + ((240-y) << 9) + x) = temp2;

			*(Video_Mem_ptr + (y << 9) + x) = temp3;
		}
	}
}

void mirror(volatile short *Video_Mem_ptr){
	int x,y;
	for (y = 0; y < 240; y++) {
		for (x = 0; x < 160; x++) {
			short temp2 = *(Video_Mem_ptr + (y << 9) + x);
			short temp3 = *(Video_Mem_ptr + (y << 9) + (320-x));

			*(Video_Mem_ptr + (y << 9) + (320-x)) = temp2;

			*(Video_Mem_ptr + (y << 9) + x) = temp3;
		}
	}
}

void invert(volatile short *Video_Mem_ptr, volatile int *KEY_ptr, volatile int * Video_In_DMA_ptr){
	blackwhite(Video_Mem_ptr);
	
	int x,y;
	while(1){
			if(*KEY_ptr == 2){
				while (*KEY_ptr != 0);
				for (y = 0; y < 240; y++) {
					for (x = 0; x < 320; x++) {
						short temp2 = *(Video_Mem_ptr + (y << 9) + x);
						temp2 = 0xffff - temp2;
						*(Video_Mem_ptr + (y << 9) + x) = temp2;
						
					}
				}
				//Invert black n white
			}
			else if(*KEY_ptr == 1){
				*(Video_In_DMA_ptr + 3)	= 0x4;
				while (*KEY_ptr != 0);
				break;
			}
		}
}

int main(void)
{
	volatile int * KEY_ptr				= (int *) KEY_BASE;
	volatile int * Video_In_DMA_ptr	= (int *) VIDEO_IN_BASE;
	volatile short * Video_Mem_ptr	= (short *) FPGA_ONCHIP_BASE;
	volatile short * Char_Mem_ptr = (short *) CHAR_BUFFER_BASE;

	int x, y;
	int count = 0;
	int task = 1;

	*(Video_In_DMA_ptr + 3)	= 0x4;				// Enable the video
	while(1){
		while (1)
		{
			if (*KEY_ptr == 1)						// check if any KEY was pressed
			{
				*(Video_In_DMA_ptr + 3) = 0x0;			// Disable the video to capture one frame
				count++;
				while (*KEY_ptr != 0);				// wait for pushbutton KEY release
				break;
			}
		}
		
		int x1 = 1;
		int y1 = 1;
		int offset = (y1<<7) + x1;
		char buf[5];
		sprintf(buf, "%d", count);
		char * text_ptr = buf;

		while (*(text_ptr)){
			*((Char_Mem_ptr + offset)) = *(text_ptr);
			
			++text_ptr;
			++offset;
		}
		
		if (task == 1){
			timestamp(Char_Mem_ptr);
		}
		else if(task == 2){
			invert(Video_Mem_ptr,KEY_ptr,Video_In_DMA_ptr);
		}
		else if(task == 3){
			flip(Video_Mem_ptr);
		}
		else if(task == 4){
			mirror(Video_Mem_ptr);
		}
		if (task != 2){
			standard(KEY_ptr, Video_In_DMA_ptr);
		}

		task++;
		if (task > 4){
			task = 1;
		}
	}
}