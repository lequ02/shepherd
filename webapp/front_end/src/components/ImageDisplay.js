import React from "react";
import { saveAs } from "file-saver";
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import ImageListItemBar from '@mui/material/ImageListItemBar';
import IconButton from '@mui/material/IconButton';
import HighlightOffIcon from '@mui/icons-material/HighlightOff';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import { Grid } from "@mui/material";

const ImageDisplay = (props) => {
    const { data, deleteImage } = props;
    const downloadImg = (src) => {
        saveAs(src, 'image.jpg')
    }
    return (
        <ImageList sx={{ width: 250, height: 450 }} cols={1} rowHeight={164}>
            {data.map((imgsrc) => (
                <ImageListItem key={`img-${Math.random() * 10000000}`}>
                    <img
                        src={imgsrc.src}
                        alt='capture'
                        loading="lazy"
                    />
                    <ImageListItemBar
                    sx={{
                        background:
                        'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, ' +
                        'rgba(0,0,0,0.3) 70%, rgba(0,0,0,0) 100%)',
                    }}
                    position="top"
                    actionIcon={
                        <Grid container>
                            <Grid item>
                                <IconButton
                                    sx={{ color: 'white' }}
                                    onClick={() => deleteImage(imgsrc.id)}
                                >
                                    <HighlightOffIcon />
                                </IconButton>
                            </Grid>
                            <Grid item>
                                <IconButton
                                    sx={{ color: 'white' }}
                                    onClick={() => downloadImg(imgsrc.src)}
                                >
                                    <CloudDownloadIcon />
                                </IconButton>
                            </Grid>
                        </Grid>
                    }
                    actionPosition="left"
                    />
                </ImageListItem>
            ))}
        </ImageList>
    )
};

export default ImageDisplay;
