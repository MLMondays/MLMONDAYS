/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

function Help(props) {
  const {config: siteConfig, language = ''} = props;
  const {baseUrl, docsUrl} = siteConfig;
  const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
  const langPart = `${language ? `${language}/` : ''}`;
  const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

  const supportLinks = [
    {
      content: `Learn more using the [documentation on this site.](${docUrl(
        'doc1.html',
      )})`,
      title: 'Browse Docs',
    },
    {
      content: 'Ask questions about the documentation and project on the dedicated Microsoft Teams channel - '+
      'please check back later for more details',
      title: 'Join the community',
    },
    {
      content: 'This website is your go-to for all information about this course and will be updated regularly. '+
      'Find out what is new with this project by checking this site regularly in the weeks leading up to the course '+
      'and during the course.<br/><br/> Also, check out the Blog posts.',
      title: 'Stay up to date',
    },
  ];

  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer documentContainer postContainer">
        <div className="post">
          <header className="postHeader">
            <h1>Need help?</h1>
          </header>
          <p>This project is maintained by a dedicated group of people, but we are also reliant on course participants to be active on the forums and message boards, helping each other out as much as possible. Please check back later for more details.</p>
          <GridBlock contents={supportLinks} layout="threeColumn" />
        </div>
      </Container>
    </div>
  );
}

module.exports = Help;
