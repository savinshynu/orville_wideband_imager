# orville_wideband_imager
The Orville Wideband Imager

Requires a change in `bifrost/src/formats/cor.hpp` of

```
pkt->chan0 = be16toh(pkt_hdr->first_chan) \
             - nchan_pkt * (server - 1);
```

to

```
pkt->chan0 = be16toh(pkt_hdr->first_chan) \
             - 4*nchan_pkt * (server - 1);
```
