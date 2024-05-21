# Flow

1. Output model to the format of lzma
2. Make RTL from python3 make_rtl
3. Synthesize with Vivado


# To Open Vivado

Make a symlink to the missing library
```bash
ln -s /cluster/apps/eb/software/ncurses/6.2-GCCcore-10.2.0/lib/libtinfo.so.6 libtinfo.so.5
```

Create a new library path, inside the directory, you have put the symlink
```bash
export LD_LIBRARY_PATH=/cluster/home/charalab/vivado:$LD_LIBRARY_PATH
```

It seems that you need a "LD_LIBRARY_PATH"!   
Just "$PATH" is not working.


# Training

```bash
```


# Information

Input size of mnist is 4700
While the Audio is 64000