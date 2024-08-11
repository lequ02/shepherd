import React, { useRef, useEffect, useState, useCallback } from "react";
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import DrawRect from "./helpers/DrawRect";
import Grid from '@mui/material/Grid';
import Button from '@mui/material/Button';
import withStyles from '@mui/styles/withStyles';
import ImageDisplay from "./ImageDisplay";

const styles = () => ({
    webcam: {
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,
        textAlign: "center",
        zindex: 9,
    },
    canvas: {
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,
        textAlign: "center",
        zindex: 8,
    },
    cameraContainer: {
        position: "relative",
    },
    actionButton: {
        paddingBottom: '16px',
    },
    webcamContainer: {
        position: "relative",
    },
    imageBody: {
        marginTop: '30px !important',
        marginBottom: '16px !important'
    }
});

const WebCam = (props) => {
    const {
        classes, setObjectData, setOpenStatistic, setImgSrc, imgSrc,
    } = props;
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const [openCamera, setOpenCamera] = useState(false);
    const handleClickCamera = () => {
        setOpenCamera(!openCamera);
        setOpenStatistic();
        setObjectData([]);
        setImgSrc([]);
    }
    const deleteImage = (imgId) => {
        const newImgList = imgSrc.filter(img => img.id !== imgId);
        setImgSrc(newImgList);
    }
    const runCoco = async () => {
        const net = await cocossd.load();
        setInterval(() => {
            detect(net);
        }, 1000);
    };
    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setImgSrc([
            ...imgSrc,
            {
                id: `img-${Math.random() * 100000000}`,
                src: imageSrc,
            }
        ])
      }, [webcamRef, setImgSrc, imgSrc]);

    const detect = async (net) => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video.readyState === 4
        ) {
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            canvasRef.current.width = videoWidth;
            canvasRef.current.height = videoHeight;

            const obj = await net.detect(video);
            setObjectData(obj);

            const ctx = canvasRef.current.getContext("2d");
            const canvasInfo = {
                top: canvasRef.current.getBoundingClientRect().top,
                left: canvasRef.current.getBoundingClientRect().left,
                right: canvasRef.current.getBoundingClientRect().right,
                bottom: canvasRef.current.getBoundingClientRect().bottom,
            }
            DrawRect(obj, ctx, canvasInfo);
        }
    };

    useEffect(() => {
        runCoco();
    }, []);

    return (
    <Grid container direction="column">
        {openCamera ? (
        <Grid container>
            <Grid item xs={2} container direction="column">
                <Grid item className={classes.actionButton}>
                    <Button variant="contained" onClick={handleClickCamera}>Stop Camera</Button>
                </Grid>
                <Grid item className={classes.actionButton}>
                    <Button variant="contained" onClick={capture}>Capture Picture</Button>
                </Grid>
            </Grid>
            <Grid item xs={7} className={classes.webcamContainer}>
                <>
                    <Webcam
                        ref={webcamRef}
                        muted={true}
                        screenshotFormat="image/jpeg"
                        className={classes.webcam}
                    />

                    <canvas
                        ref={canvasRef}
                        className={classes.canvas}
                    />
                </>
            </Grid>
            <Grid item xs={3}>
                <ImageDisplay data={imgSrc} deleteImage={deleteImage} />
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
                <Grid item>
                    <Button variant="contained" onClick={handleClickCamera}>Open Camera</Button>
                </Grid>
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

export default withStyles(styles)(WebCam);
