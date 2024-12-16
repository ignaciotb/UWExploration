std_msgs/Header header
uint8   type          # Packet Type, 0xE2 = Sonar 8 Bit
uint32  time          # Ping time   [100 us]
uint8   frequency_id  # Freq Id, currently 0x00
int16   gain
uint16  decimation
float32 max_duration  # Max travel time of outermost bins (s)
uint8[] port_channel
uint8[] starboard_channel
uint8[] extra_channel
