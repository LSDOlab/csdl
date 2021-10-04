import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
    {
        title: 'Automatically compute derivatives',
        Svg: require('/img/derivatives.svg').default,
        throwIfNamespace: false,
        description: (
            <>
                Trust CSDL's fully intrusive design to compute exact
                derivatives for each operation with no additional code
                required.
            </>
        ),
    },
    // {
    //     title: 'Work at a high level of abstraction',
    //     Svg: require('/img/undraw_rocket.svg').default,
    //     description: (
    //         <>
    //             The CSDL compiler builds an intermediate representation of
    //             your model, performing implementation independent code
    //             optimizations, letting you focus on modeling, and not
    //             low level implementation.
    //         </>
    //     ),
    // },
    {
        title: 'Work with large scale systems',
        Svg: require('/img/undraw_aircraft.svg').default,
        description: (
            <>
                CSDL relies on the Modular Analysis and Unified
                Derivatives (MAUD) architecture, enabling efficient
                derivative computation for large scale systems, even
                when external solvers are used for model evaluation.
            </>
        ),
    },
    {
        title: 'Get off the ground quickly',
        Svg: require('/img/undraw_rocket.svg').default,
        description: (
            <>
                Focus on modeling physical systems, not implementing
                algorithms.
                Use a functional and/or object oriented style to define
                system models, and take advantage of CSDL's early,
                helpful error messages to build correct model
                specifications.
            </>
        ),
    },
    // {
    //     title: 'Control compile time execution',
    //     Svg: require('/img/Python_Outline.svg').default,
    //     description: (
    //         <>
    //             Harness the full power of Python to control compile time
    //             behavior, and use parameters to make model definitions
    //             more generic and reusable.
    //         </>
    //     ),
    // },
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
        <div className="container">
            <div className="row">
                {
                    FeatureList.map((props, idx) => (<
                        Feature key={idx} {...props}
                    />
                    ))
                }
            </div>
        </div>
    </section>
    );
}
