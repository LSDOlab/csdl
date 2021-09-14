import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
    {
        title: 'Easy to Use',
        Svg: require('../../static/img/undraw_drone.svg').default,
        description: (
            <>Use CSDL in any project.</>
        ),
    },
    {
        title: 'Work with large scale systems',
        Svg: require('../../static/img/undraw_aircraft.svg').default,
        description: (
            <>CSDL automates derivative computation across multiple disciplines without any overhead imposed on the user, making project code easy to write, easy to read, and easy to maintain.</>
        ),
    },
    {
        title: 'Helpful Error messages',
        Svg: require('../../static/img/undraw_rocket.svg').default,
        description: (
            <>CSDL has early, helpful error messages to prevent users from making modeling errors early in the design process.</>
        ),
    },
];

function Feature({ Svg, title, description }) {
    return (
        <div className={clsx('col col--4')} >
            <div className="text--center" >
                <Svg className={styles.featureSvg} alt={title} />
            </div>
            <div className="text--center padding-horiz--md">
                <h3> {title} </h3>
                <p> {description} </p>
            </div>
        </div>
    );
}

export default function HomepageFeatures() {
    return (<
        section className={styles.features} >
        <
        div className="container" >
            <
        div className="row" > {
                    FeatureList.map((props, idx) => (<
                        Feature key={idx} {...props}
                    />
                    ))
                } <
        /div> < /
        div > <
        /section>
                );
}
