import React, { useRef, useState, useEffect } from "react";
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import Grid from '@mui/material/Grid';
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import withStyles from '@mui/styles/withStyles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import Statistic from "./Statistic";
import ChartStatistic from "./charStatistic";
import axios from 'axios';

const styles = () => ({
    video: {
        left: 0,
        right: 0,
        textAlign: "center",
        width: '500px',
        height: '500px',
    },
    actionButton: {
        paddingBottom: '16px',
    },
    div: {
        height: '500px',
    },
    imageBody: {
        marginTop: '30px !important',
        marginBottom: '16px !important'
    }
});

const VisuallyHiddenInput = styled('input')({
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: 1,
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    whiteSpace: 'nowrap',
    width: 1,
  });
  

const VideoInput = (props) => {
    const {
        classes, setObjectData, setOpenStatistic, objectData
    } = props;
    const [isTracking, setIsTracking] = useState(false);
    const videoRef = useRef(null);
    const [openVideo, setOpenVideo] = useState(false);
    const [videoSrc, setVideoSrc] = useState('');
    const handleClickVideo = () => {
        setOpenVideo(!openVideo);
        setOpenStatistic();
        setObjectData({});
    }
    const handleUpload = (event) => {
        setOpenStatistic()
        setOpenVideo(true);
        const file = event.target.files[0];
        const objectUrl = URL.createObjectURL(file);
        setVideoSrc(objectUrl);
        const formData = new FormData();
        formData.append("file", file);
        setIsTracking(true);
        axios({
            method: 'post',
            url: 'http://127.0.0.1:5000/upload_video',
            data: formData
          }).then((res) => {
            setIsTracking(false);
          }).catch(({ response }) => {
            console.log('error upload data');
          });;
    };

    useEffect(() => {
        if (isTracking) {
          const intervalId = setInterval(() => {
            console.log('running fetch: ');
            axios({
                method: 'get',
                url: 'http://127.0.0.1:5000/get_current_statistic',
              }).then((res) => {
                setObjectData(res.data.statistic_data);
              }).catch((error) => {
                console.log(error);
              });;
          }, 2000);
          return () => clearInterval(intervalId);
        }
      }, [isTracking]);

    return (
    <Grid container direction="column">
        {openVideo ? (
        <Grid container justifyContent="center" alignItems="center">
            <Grid item container direction="column" xs={7} className={classes.webcamContainer}>
                <Grid item>
                    <video ref={videoRef} muted={true} controls autoPlay className={classes.video}>
                        <source src={videoSrc} type="video/mp4" />
                    </video>
                </Grid>
                <Grid item>
                    <Button disabled={isTracking} variant="contained" onClick={handleClickVideo}>Stop Video</Button>
                </Grid>
            </Grid>
            <Grid item>
                <ChartStatistic data={objectData}/>
            </Grid>
        </Grid>
        ) : (
            <Grid item container direction="column" justifyContent="center" alignItems="center">
                <Grid item className={classes.imageBody}>
                    <img
                        src="/logo.png"
                        alt="logoimage"
                        width='200'
                        height='100'
                    />
                </Grid>
                <Grid item>
                    <Button
                        component="label"
                        role={undefined}
                        variant="contained"
                        tabIndex={-1}
                        startIcon={<CloudUploadIcon />}
                    >
                        Upload file
                        <VisuallyHiddenInput type="file" accept="video/*" onChange={handleUpload}/>
                    </Button>
                </Grid>
                <Grid item className={classes.imageBody}>
                    <img
                        src="/word.png"
                        alt="logoimage"
                        width='200'
                        height='100'
                    />
                </Grid>
            </Grid>
        )}
    </Grid>
    );
}

export default withStyles(styles)(VideoInput);
