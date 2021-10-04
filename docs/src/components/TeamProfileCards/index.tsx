/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { ReactNode } from 'react';
import Translate from '@docusaurus/Translate';
import Link from '@docusaurus/Link';

function WebsiteLink({ to, children }: { to: string; children?: ReactNode }) {
    return (
        <Link to={to}>
            {children || (
                <Translate id="team.profile.websiteLinkLabel">website</Translate>
            )}
        </Link>
    );
}

interface ProfileProps {
    className?: string;
    name: string;
    children: ReactNode;
    githubUrl?: string;
    twitterUrl?: string;
}

function TeamProfileCard({
    className,
    name,
    children,
    githubUrl,
    twitterUrl,
}: ProfileProps) {
    return (
        <div className={className}>
            <div className="card card--full-height">
                <div className="card__header">
                    <div className="avatar avatar--vertical">
                        <img
                            className="avatar__photo avatar__photo--xl"
                            src={githubUrl + '.png'}
                            alt={`${name}'s avatar`}
                        />
                        <div className="avatar__intro">
                            <h3 className="avatar__name">{name}</h3>
                        </div>
                    </div>
                </div>
                <div className="card__body">{children}</div>
                <div className="card__footer">
                    <div className="button-group button-group--block">
                        {githubUrl && (
                            <a className="button button--secondary" href={githubUrl}>
                                GitHub
                            </a>
                        )}
                        {twitterUrl && (
                            <a className="button button--secondary" href={twitterUrl}>
                                Twitter
                            </a>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

function TeamProfileCardCol(props: ProfileProps) {
    return (
        <TeamProfileCard {...props} className={'col col--6 margin-bottom--lg'} />
    );
}

export function CoreTeamRow() {
    return (
        <div className="row">
            <TeamProfileCardCol
                name="Victor Gandarillas"
                githubUrl="https://github.com/vgucsd"
            // twitterUrl="https://twitter.com/"
            >
                <Translate id="team.profile.Victor Gandarillas.body">
                    Creator of CSDL, CSDL compiler frontend, and CSDL-OM
                    compiler back end
                </Translate>
            </TeamProfileCardCol>
        </div>
    );
}

export function StdLibTeamRow() {
    return (
        <div className="row">
            <TeamProfileCardCol
                name="Victor Gandarillas"
                githubUrl="https://github.com/vgucsd"
            // twitterUrl="https://twitter.com/"
            >
                <Translate id="team.profile.Victor Gandarillas.body">
                    Creator of CSDL, CSDL compiler frontend, and CSDL-OM
                    compiler back end
                </Translate>
            </TeamProfileCardCol>
            <TeamProfileCardCol
                name="Anugrah Joshy"
                githubUrl="https://github.com/anugrahjo"
            // twitterUrl="https://twitter.com/"
            >
                <Translate id="team.profile.Anugrah Joshy.body">
                    Standard Library Contributor
                </Translate>
            </TeamProfileCardCol>
            <TeamProfileCardCol
                name="Alex Ivanov"
                githubUrl="https://github.com/AlexKIvanov"
            // twitterUrl="https://twitter.com/"
            >
                <Translate
                    id="team.profile.Sebastien Lorber.body">
                    Standard Library Contributor
                </Translate>
            </TeamProfileCardCol>
        </div>
    );
}
