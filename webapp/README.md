# Pi515 Challenge - Team Norse

Monitoring Farm Sheep Movement with Deep Learning and Object Tracking


## Installation

### Install and setup git
[https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### Install NodeJS
[https://nodejs.org/en/download](https://nodejs.org/en/download)

Check Node Version
```
node --version
```

### Install Yarn
```
npm install --global yarn
```
Check Yarn version
```
yarn --version
```

### Install Python
[https://www.python.org/downloads/](https://www.python.org/downloads/)

## Setup Development Environment

1. Clone git repo to your local machine
```
cd path_to_location_you_want_to_store_the_project_in_your_local_machine
git clone https://github.com/vuongtr01/sheeps_tracking.git
```

2. Move to the project directory
```
cd path_to_project_in_your_local_machine
```

3. Install Front End dependencies
```
cd front_end
yarn install
```

4. Install Back End virtual environment and dependencies

Install virtualenv library
```
pip3 install virtualenv
```
Create virtual environment
```
python3 -m venv venv
```
Activate virtual environment
```
source venv/bin/activate 
```
Install dependencies
```
pip3 install -r requirements.txt
```

Deactivate virtual environment
```
deactivate
```
## Runs the app in the development mode
1. Start back end server

Open a terminal:
```
cd back_end
flask run
```

2. Start front end server

Open another terminal
```
cd front_end
yarn start
```
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.
