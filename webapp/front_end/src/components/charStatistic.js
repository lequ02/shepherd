import React from "react";
import withStyles from '@mui/styles/withStyles';
import { BarChart } from '@mui/x-charts/BarChart';

const ChartStatistic = (props) => {
    const { data } = props;

    return (
        <BarChart
            xAxis={[
                {
                id: 'barCategories',
                data: Object.keys(data),
                scaleType: 'band',
                label: 'Sheep Id',
                },
            ]}
            series={[
                {
                    data: Object.values(data),
                },
            ]}
            yAxis={
                [{
                    label: 'Distance Moved (meters)'
                }]
            }
            width={550}
            height={300}
        />
    )
}

export default ChartStatistic;