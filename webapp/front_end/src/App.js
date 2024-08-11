import React, { useState } from "react";
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import TabContext from '@mui/lab/TabContext';
import TabList from '@mui/lab/TabList';
import TabPanel from '@mui/lab/TabPanel';
import withStyles from '@mui/styles/withStyles';
import VideoInput from "./components/VideoInput";
import WebCam from "./components/WebCam";
import Statistic from "./components/Statistic";

const styles = () => ({
    statistic: {
        paddingTop: '32px',
    }
})

const App = (props) => {
    const { classes } = props;
    const [value, setValue] = useState('1');
    const [objectData, setObjectData] = useState({});
    const [openStatistic, setOpenStatistic] = useState(false);
    const [imgSrc, setImgSrc] = useState([]);
    const handleChange = (_, newValue) => {
        setValue(newValue);
        setOpenStatistic(false);
    };
    return (
    <Box sx={{width: '100%', typography: 'body1' }}>
        <TabContext value={value}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Grid container>
                    <Grid item xs={3}>
                        <img
                            src="/logo.png"
                            alt="logoimage"
                            width='100'
                            height='50'
                        />
                    </Grid>
                    <Grid item xs={2}/>
                    <Grid item xs={3}>
                        <TabList onChange={handleChange} aria-label="lab API tabs example">
                            <Tab label="Webcam" value="1" />
                            <Tab label="Video Input" value="2" />
                        </TabList>
                    </Grid>  
                </Grid>
            </Box>
            <Grid container direction="column">
                <Grid item>
                    <Grid container direction="column">
                        <Grid item />
                        <Grid item>
                            <TabPanel value="1">
                                <WebCam
                                    setObjectData={setObjectData}
                                    objectData={objectData}
                                    setOpenStatistic={() => setOpenStatistic(!openStatistic)}
                                    setImgSrc={setImgSrc}
                                    imgSrc={imgSrc}
                                />
                            </TabPanel>
                            <TabPanel value="2">
                                <VideoInput
                                    setObjectData={setObjectData}
                                    objectData={objectData}
                                    setOpenStatistic={() => setOpenStatistic(!openStatistic)}
                                    openStatistic={openStatistic}
                                />
                            </TabPanel>
                        </Grid>
                        <Grid item />
                    </Grid>
                </Grid>
            </Grid>
        </TabContext>
    </Box>
    )
}

export default withStyles(styles)(App);
