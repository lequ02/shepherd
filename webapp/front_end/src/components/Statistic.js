import React from "react";
import withStyles from '@mui/styles/withStyles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

const styles = () => ({
    headCell: {
        backgroundColor: '#0a0101',
        color: 'white !important',
    },
    tableRow: {
        '&:nth-of-type(odd)': {
            backgroundColor: '#EFEEE7',
          },
          // hide last border
          '&:last-child td, &:last-child th': {
            border: 0,
          },
    }
});

const Statistic = (props) => {
    const { classes, data } = props;
    return (
        <TableContainer component={Paper} sx={{ maxHeight: 250 }}>
            <Table aria-label="simple table">
                <TableHead>
                    <TableRow>
                        <TableCell className={classes.headCell}>Sheep ID</TableCell>
                        <TableCell className={classes.headCell} align="right">Distance moved (meters)</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {Object.keys(data).map((row) => (
                        <TableRow
                            className={classes.tableRow}
                            key={`object-${Math.random() * 1000000000}`}
                        >
                            <TableCell component="th" scope="row">
                                {parseInt(row)}
                            </TableCell>
                            <TableCell align="right">{data[row]}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    )
};

export default withStyles(styles)(Statistic);
