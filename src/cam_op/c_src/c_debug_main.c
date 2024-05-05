#include "mv_cam.h"
#include <stdlib.h>

int main()
{
    uint8_t status = initialize_camera(1);
    if (status != CAMERA_STATUS_SUCCESS)
    {
        printf("Error code: %d\n", status);
        return 1;
    }
    uint8_t *buffer = (uint8_t *)malloc(2048 * 2048 * 3 * sizeof(uint8_t));
    uint32_t width, height;
    get_image(0, buffer, &width, &height, 0);
    printf("Image width: %d\n", width);
    printf("Image height: %d\n", height);
    uninitialize_camera();

    return 0;
}